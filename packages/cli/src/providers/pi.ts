import { type ChildProcess, execSync, spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { ToolName } from "@agent-kanban/shared";
import { createLogger } from "../logger.js";
import type { AgentEvent, AgentHandle, AgentProvider, ContentBlock, ExecuteOpts, HistoryEvent, RuntimeModel, UsageInfo } from "./types.js";

const logger = createLogger("pi");
const DEFAULT_TOOLS = "read,bash,edit,write,grep,find,ls";
const RESPONSE_TIMEOUT_MS = 10_000;

interface PendingResponse {
  reject(error: Error): void;
  resolve(response: PiResponse): void;
  timer: ReturnType<typeof setTimeout>;
}

interface PiResponse {
  command?: string;
  data?: unknown;
  error?: string;
  id?: string;
  success?: boolean;
  type: "response";
}

export interface PiMapState {
  cumulativeCost: number;
  resultSeen: boolean;
  pendingTools: Map<string, ContentBlock & { type: "tool_use" }>;
}

interface PiRpcState {
  aborted: boolean;
  stderr: string;
}

export function buildPiArgs(opts: ExecuteOpts): string[] {
  const args = ["--mode", "rpc", "--approve", "--no-extensions", "--tools", DEFAULT_TOOLS, "--name", `ak-${opts.sessionId}`];
  if (opts.resume) {
    if (!opts.resumeToken) throw new Error("pi: resume requested but no resumeToken provided");
    args.push("--session", opts.resumeToken);
  }
  if (opts.model) args.push("--model", opts.model);
  const systemPrompt = readSystemPrompt(opts.systemPromptFile);
  if (systemPrompt) args.push("--system-prompt", systemPrompt);
  return args;
}

export function parsePiModelList(output: string): RuntimeModel[] {
  const lines = output
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  const rows = lines[0]?.startsWith("provider") ? lines.slice(1) : lines;
  const models: RuntimeModel[] = [];

  for (const row of rows) {
    const columns = row.split(/\s{2,}/).filter(Boolean);
    if (columns.length < 2) continue;
    const [provider, model, context, maxOut, thinking, images] = columns;
    models.push({
      id: `${provider}/${model}`,
      name: model,
      description: provider,
      context_window: parseTokenCount(context),
      output_token_limit: parseTokenCount(maxOut),
      supports: {
        images: images === "yes",
        thinking: thinking === "yes",
      },
    });
  }

  return models;
}

export function mapPiEvent(event: unknown, state: PiMapState): AgentEvent[] {
  if (!isRecord(event)) return [];
  const type = event.type;

  if (type === "turn_start") return [{ type: "turn.start" }];

  if (type === "message_update") {
    return mapAssistantDelta(event.assistantMessageEvent, state);
  }

  if (type === "tool_execution_start") {
    const toolCallId = stringValue(event.toolCallId) ?? `pi-tool-${Date.now()}`;
    const toolName = stringValue(event.toolName) ?? "tool";
    const block: ContentBlock & { type: "tool_use" } = {
      type: "tool_use",
      id: toolCallId,
      name: normalizePiToolName(toolName),
      input: normalizePiToolInput(toolName, isRecord(event.args) ? event.args : {}),
    };
    state.pendingTools.set(toolCallId, block);
    return [{ type: "block.start", block }];
  }

  if (type === "tool_execution_end") {
    const toolCallId = stringValue(event.toolCallId) ?? "";
    const events: AgentEvent[] = [];
    const pending = state.pendingTools.get(toolCallId);
    if (pending) {
      events.push({ type: "block.done", block: pending });
      state.pendingTools.delete(toolCallId);
    }
    events.push({
      type: "block.done",
      block: {
        type: "tool_result",
        tool_use_id: toolCallId,
        output: flattenToolResult(event.result),
        error: event.isError === true ? true : undefined,
      },
    });
    return events;
  }

  if (type === "turn_end") {
    const usage = usageFromTurnEnd(event);
    state.cumulativeCost += usage.cost;
    state.resultSeen = true;
    return [{ type: "turn.end", cost: state.cumulativeCost, usage: usage.tokens }];
  }

  if (type === "agent_end") {
    if (state.resultSeen) return [];
    state.resultSeen = true;
    return [{ type: "turn.end", cost: state.cumulativeCost }];
  }

  if (type === "extension_error") {
    return [{ type: "turn.error", detail: textValue(event.error) || JSON.stringify(event) }];
  }

  return [];
}

export const piProvider: AgentProvider = {
  name: "pi",
  label: "Pi",

  async checkAvailability() {
    try {
      const models = await this.listModels!();
      if (models.length === 0) return { status: "unauthorized" as const, detail: "Pi has no authenticated model provider" };
      return { status: "ready" as const };
    } catch (err) {
      return { status: "unhealthy" as const, detail: `Pi model listing failed: ${errMessage(err)}` };
    }
  },

  async execute(opts: ExecuteOpts): Promise<AgentHandle> {
    return startPiRpc(opts);
  },

  async fetchUsage(): Promise<UsageInfo | null> {
    return null;
  },

  async getHistory(_sessionId: string, _resumeToken?: string): Promise<HistoryEvent[]> {
    return [];
  },

  async listModels(): Promise<RuntimeModel[]> {
    const output = execSync("pi --list-models", { encoding: "utf-8", stdio: ["ignore", "pipe", "pipe"] });
    return parsePiModelList(output);
  },
};

async function startPiRpc(opts: ExecuteOpts): Promise<AgentHandle> {
  const proc = spawn("pi", buildPiArgs(opts), {
    cwd: opts.cwd,
    env: opts.env,
    stdio: ["pipe", "pipe", "pipe"],
  });

  const queue = new EventQueue();
  const mapState: PiMapState = { cumulativeCost: 0, resultSeen: false, pendingTools: new Map() };
  const runtimeState: PiRpcState = { aborted: false, stderr: "" };
  const pending = new Map<string, PendingResponse>();
  let resumeToken = opts.resumeToken;
  let requestSeq = 0;

  const sendRaw = (payload: Record<string, unknown>) => {
    if (!proc.stdin || proc.stdin.destroyed) throw new Error("pi stdin is closed");
    proc.stdin.write(`${JSON.stringify(payload)}\n`);
  };

  const rejectPending = (error: Error) => {
    for (const entry of pending.values()) {
      clearTimeout(entry.timer);
      entry.reject(error);
    }
    pending.clear();
  };

  const request = (payload: Record<string, unknown>, timeoutMs = RESPONSE_TIMEOUT_MS): Promise<PiResponse> => {
    const id = `ak-${++requestSeq}`;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        pending.delete(id);
        reject(new Error(`pi RPC ${payload.type ?? "request"} timed out`));
      }, timeoutMs);
      pending.set(id, { resolve, reject, timer });
      try {
        sendRaw({ ...payload, id });
      } catch (err) {
        clearTimeout(timer);
        pending.delete(id);
        reject(err instanceof Error ? err : new Error(String(err)));
      }
    });
  };

  attachOutput(proc, queue, pending, mapState, runtimeState, rejectPending, sendRaw);

  const stateResponse = await request({ type: "get_state" });
  if (stateResponse.success === false) throw new Error(stateResponse.error || "pi get_state failed");
  resumeToken = resumeTokenFromState(stateResponse.data) ?? resumeToken;

  request({ type: "prompt", message: opts.taskContext })
    .then((response) => {
      if (response.success === false) {
        queue.push({ type: "turn.error", detail: response.error || "Pi rejected prompt" });
        queue.finish();
      }
    })
    .catch((err) => {
      queue.finish(err);
    });

  return {
    events: queue.iterate(),
    async abort() {
      if (runtimeState.aborted) return;
      runtimeState.aborted = true;
      try {
        sendRaw({ type: "abort" });
      } catch {
        // Process may already be gone.
      }
      queue.finish();
      rejectPending(new Error("pi RPC aborted"));
      await terminateProcess(proc);
    },
    async send(message: string) {
      if (runtimeState.aborted || queue.done) return;
      await request({ type: "prompt", message, streamingBehavior: "steer" });
    },
    getResumeToken() {
      return resumeToken;
    },
  };
}

function attachOutput(
  proc: ChildProcess,
  queue: EventQueue,
  pending: Map<string, PendingResponse>,
  mapState: PiMapState,
  runtimeState: PiRpcState,
  rejectPending: (error: Error) => void,
  sendRaw: (payload: Record<string, unknown>) => void,
): void {
  let buffer = "";

  proc.stdout?.on("data", (chunk: Buffer) => {
    buffer += chunk.toString();
    while (true) {
      const index = buffer.indexOf("\n");
      if (index === -1) break;
      const line = buffer.slice(0, index).replace(/\r$/, "");
      buffer = buffer.slice(index + 1);
      handleLine(line, queue, pending, mapState, sendRaw, proc, runtimeState);
    }
  });

  proc.stderr?.on("data", (chunk: Buffer) => {
    runtimeState.stderr += chunk.toString();
    if (runtimeState.stderr.length > 50_000) runtimeState.stderr = runtimeState.stderr.slice(-25_000);
  });

  proc.on("error", (err) => {
    rejectPending(err);
    queue.finish(err);
  });

  proc.on("close", (code) => {
    if (buffer.trim()) handleLine(buffer.replace(/\r$/, ""), queue, pending, mapState, sendRaw, proc, runtimeState);
    if (queue.done || runtimeState.aborted) return;
    if (code === 0) {
      queue.finish();
      return;
    }
    const stderr = runtimeState.stderr.trim().split("\n").slice(-10).join("\n");
    queue.finish(new Error(`pi exited with code ${code}${stderr ? `: ${stderr}` : ""}`));
  });
}

function handleLine(
  line: string,
  queue: EventQueue,
  pending: Map<string, PendingResponse>,
  mapState: PiMapState,
  sendRaw: (payload: Record<string, unknown>) => void,
  proc: ChildProcess,
  runtimeState: PiRpcState,
): void {
  if (!line.trim()) return;
  const event = parseJsonObject(line);
  if (!event) {
    logger.debug(`Ignoring non-JSON pi output: ${line}`);
    return;
  }

  if (event.type === "response") {
    const id = stringValue(event.id);
    if (id) {
      const waiter = pending.get(id);
      if (waiter) {
        clearTimeout(waiter.timer);
        pending.delete(id);
        waiter.resolve(piResponseFrom(event));
      }
    }
    return;
  }

  if (event.type === "extension_ui_request") {
    handleExtensionUiRequest(event, sendRaw);
    return;
  }

  for (const mapped of mapPiEvent(event, mapState)) queue.push(mapped);

  if (event.type === "agent_end") {
    queue.finish();
    runtimeState.aborted = true;
    terminateProcess(proc).catch((err) => logger.warn(`Failed to terminate pi after completion: ${errMessage(err)}`));
  }
}

function piResponseFrom(event: Record<string, unknown>): PiResponse {
  return {
    type: "response",
    command: stringValue(event.command) ?? undefined,
    data: event.data,
    error: stringValue(event.error) ?? undefined,
    id: stringValue(event.id) ?? undefined,
    success: typeof event.success === "boolean" ? event.success : undefined,
  };
}

function handleExtensionUiRequest(event: Record<string, unknown>, sendRaw: (payload: Record<string, unknown>) => void): void {
  const id = stringValue(event.id);
  const method = stringValue(event.method);
  if (!id) return;
  if (method === "confirm") {
    sendRaw({ type: "extension_ui_response", id, confirmed: false });
    return;
  }
  if (method === "select" || method === "input" || method === "editor") {
    sendRaw({ type: "extension_ui_response", id, cancelled: true });
  }
}

function mapAssistantDelta(value: unknown, state: PiMapState): AgentEvent[] {
  if (!isRecord(value)) return [];
  const type = value.type;
  if (type === "text_start") return [{ type: "block.start", block: { type: "text", text: "" } }];
  if (type === "text_end") {
    const text = textValue(value.content) || textValue(value.text) || textFromPartial(value.partial);
    return text ? [{ type: "block.done", block: { type: "text", text } }] : [];
  }
  if (type === "thinking_start") return [{ type: "block.start", block: { type: "thinking", text: "" } }];
  if (type === "thinking_end") {
    const text = textValue(value.content) || textValue(value.thinking) || textValue(value.text) || textFromPartial(value.partial);
    return text ? [{ type: "block.done", block: { type: "thinking", text } }] : [];
  }
  if (type === "error") {
    state.resultSeen = true;
    return [{ type: "turn.error", detail: textValue(value.reason) || textValue(value.error) || JSON.stringify(value) }];
  }
  return [];
}

function normalizePiToolName(name: string): string {
  switch (name) {
    case "bash":
      return ToolName.Bash;
    case "read":
    case "ls":
      return ToolName.Read;
    case "edit":
      return ToolName.Edit;
    case "write":
      return ToolName.Write;
    case "grep":
      return ToolName.Grep;
    case "find":
      return ToolName.Glob;
    default:
      return name;
  }
}

function normalizePiToolInput(name: string, input: Record<string, unknown>): Record<string, unknown> {
  switch (name) {
    case "bash":
      return { command: String(input.command ?? "") };
    case "read":
    case "ls":
      return { filePath: String(input.path ?? input.filePath ?? "") };
    case "edit":
      return {
        filePath: String(input.path ?? input.filePath ?? ""),
        oldString: String(input.oldText ?? input.old_string ?? input.oldString ?? ""),
        newString: String(input.newText ?? input.new_string ?? input.newString ?? ""),
      };
    case "write":
      return { filePath: String(input.path ?? input.filePath ?? ""), content: String(input.content ?? "") };
    case "grep":
      return { pattern: String(input.pattern ?? ""), path: stringValue(input.path) ?? undefined };
    case "find":
      return { pattern: String(input.pattern ?? ""), path: stringValue(input.path) ?? undefined };
    default:
      return input;
  }
}

function usageFromTurnEnd(event: Record<string, unknown>): { cost: number; tokens: Record<string, number | undefined> | undefined } {
  const message = isRecord(event.message) ? event.message : event;
  const usage = isRecord(message.usage) ? message.usage : undefined;
  if (!usage) return { cost: 0, tokens: undefined };
  const cost = isRecord(usage.cost) ? (numberValue(usage.cost.total) ?? 0) : 0;
  return {
    cost,
    tokens: {
      input_tokens: numberValue(usage.input),
      output_tokens: numberValue(usage.output),
      cache_read_input_tokens: numberValue(usage.cacheRead),
      cache_creation_input_tokens: numberValue(usage.cacheWrite),
    },
  };
}

function flattenToolResult(result: unknown): string {
  if (typeof result === "string") return result;
  if (!isRecord(result)) return result == null ? "" : JSON.stringify(result);
  if (typeof result.output === "string") return result.output;
  if (Array.isArray(result.content)) {
    const texts = result.content.map(textFromContentPart).filter(Boolean);
    if (texts.length > 0) return texts.join("\n");
  }
  return JSON.stringify(result);
}

function textFromContentPart(part: unknown): string {
  if (typeof part === "string") return part;
  if (!isRecord(part)) return "";
  return textValue(part.text) || textValue(part.content) || "";
}

function textFromPartial(partial: unknown): string {
  if (!isRecord(partial)) return "";
  const content = partial.content;
  if (!Array.isArray(content)) return textValue(partial.text) || "";
  return content.map(textFromContentPart).filter(Boolean).join("\n");
}

function resumeTokenFromState(data: unknown): string | undefined {
  if (!isRecord(data)) return undefined;
  return stringValue(data.sessionFile) ?? stringValue(data.sessionId) ?? undefined;
}

function readSystemPrompt(filePath?: string): string {
  if (!filePath) return "";
  try {
    return readFileSync(filePath, "utf-8");
  } catch {
    return "";
  }
}

function parseTokenCount(value: string | undefined): number | undefined {
  if (!value || value === "-") return undefined;
  const match = value.match(/^(\d+(?:\.\d+)?)([KMG])?$/i);
  if (!match) return undefined;
  const amount = Number(match[1]);
  const suffix = match[2]?.toUpperCase();
  const multiplier = suffix === "G" ? 1_000_000_000 : suffix === "M" ? 1_000_000 : suffix === "K" ? 1_000 : 1;
  return Math.round(amount * multiplier);
}

function parseJsonObject(line: string): Record<string, unknown> | null {
  try {
    const parsed: unknown = JSON.parse(line);
    return isRecord(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function stringValue(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function textValue(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function numberValue(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function errMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  return String(err);
}

function terminateProcess(proc: ChildProcess): Promise<void> {
  return new Promise((resolve) => {
    if (!proc.pid || proc.killed) {
      resolve();
      return;
    }
    const killTimer = setTimeout(() => {
      if (!proc.killed) proc.kill("SIGKILL");
      resolve();
    }, 5000);
    proc.once("close", () => {
      clearTimeout(killTimer);
      resolve();
    });
    proc.kill("SIGTERM");
  });
}

export class EventQueue {
  private buffer: AgentEvent[] = [];
  private error: unknown = null;
  private waiter: (() => void) | null = null;
  done = false;

  push(event: AgentEvent): void {
    if (this.done) return;
    this.buffer.push(event);
    this.waiter?.();
  }

  finish(err?: unknown): void {
    if (this.done) return;
    this.done = true;
    if (err !== undefined) this.error = err;
    this.waiter?.();
  }

  async *iterate(): AsyncIterable<AgentEvent> {
    while (true) {
      while (this.buffer.length) yield this.buffer.shift()!;
      if (this.done) break;
      await new Promise<void>((resolve) => {
        this.waiter = resolve;
      });
      this.waiter = null;
    }
    while (this.buffer.length) yield this.buffer.shift()!;
    if (this.error) throw this.error;
  }
}
