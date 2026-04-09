/**
 * Ollama Agent Runner for NanoClaw
 *
 * Drop-in replacement for runContainerAgent that calls a local Ollama instance
 * instead of spawning a Docker container with the Claude Agent SDK.
 *
 * Set AGENT_BACKEND=ollama in .env to enable.
 * Set OLLAMA_URL (default: http://localhost:11434) and OLLAMA_MODEL (default: qwen2.5:7b).
 *
 * Path mapping (mirrors the container's /workspace layout):
 *   /workspace/group/...        → groups/{folder}/...
 *   /workspace/extra/{name}/... → additionalMounts[containerPath=name].hostPath/...
 *   /workspace/global/...       → groups/global/...
 */

import { EventEmitter } from 'events';
import fs from 'fs';
import path from 'path';
import type { ChildProcess } from 'child_process';

import { GROUPS_DIR, OLLAMA_MODEL, OLLAMA_URL } from './config.js';
import type { ContainerInput, ContainerOutput } from './container-runner.js';
import { resolveGroupFolderPath, resolveGroupIpcPath } from './group-folder.js';
import { logger } from './logger.js';
import type { RegisteredGroup } from './types.js';

// ─── Ollama API types ────────────────────────────────────────────────────────

interface OllamaMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string;
  tool_calls?: OllamaToolCall[];
  name?: string; // required by some Ollama versions for role=tool
}

interface OllamaToolCall {
  function: {
    name: string;
    arguments: Record<string, unknown>;
  };
}

interface OllamaTool {
  type: 'function';
  function: {
    name: string;
    description: string;
    parameters: {
      type: 'object';
      properties: Record<string, { type: string; description?: string }>;
      required: string[];
    };
  };
}

// ─── Tools ───────────────────────────────────────────────────────────────────

function buildTools(hasNotifyOwner: boolean): OllamaTool[] {
  const tools: OllamaTool[] = [
    {
      type: 'function',
      function: {
        name: 'read_file',
        description:
          'Read the contents of a file. Use /workspace/group/ for group-specific files, /workspace/extra/main-grocery/ for the shared grocery lists and rules.',
        parameters: {
          type: 'object',
          properties: {
            path: {
              type: 'string',
              description:
                'Container-style path, e.g. /workspace/extra/main-grocery/supermarket-list.md',
            },
          },
          required: ['path'],
        },
      },
    },
    {
      type: 'function',
      function: {
        name: 'write_file',
        description: 'Write (overwrite) a file with new content.',
        parameters: {
          type: 'object',
          properties: {
            path: {
              type: 'string',
              description: 'Container-style path to write to.',
            },
            content: {
              type: 'string',
              description: 'Full new content of the file.',
            },
          },
          required: ['path', 'content'],
        },
      },
    },
    {
      type: 'function',
      function: {
        name: 'send_message',
        description:
          'Send a message to the current chat immediately (for progress updates or multi-part replies).',
        parameters: {
          type: 'object',
          properties: {
            text: { type: 'string', description: 'Message text to send.' },
          },
          required: ['text'],
        },
      },
    },
  ];

  if (hasNotifyOwner) {
    tools.push({
      type: 'function',
      function: {
        name: 'notify_owner',
        description:
          'Send a notification to the owner (main channel). Call this after updating any shared grocery list.',
        parameters: {
          type: 'object',
          properties: {
            text: {
              type: 'string',
              description: 'Notification text to send to the owner.',
            },
          },
          required: ['text'],
        },
      },
    });
  }

  return tools;
}

// ─── Path resolution ─────────────────────────────────────────────────────────

/**
 * Translate a container-style path to a real host path.
 * Returns null if the path is outside allowed locations.
 */
function resolveGuestPath(
  guestPath: string,
  group: RegisteredGroup,
): string | null {
  const groupDir = resolveGroupFolderPath(group.folder);
  const globalDir = path.join(GROUPS_DIR, 'global');

  const normalize = (p: string) => p.replace(/^~/, process.env.HOME || '');

  if (guestPath === '/workspace/group' || guestPath.startsWith('/workspace/group/')) {
    const sub = guestPath.slice('/workspace/group'.length);
    return path.join(groupDir, sub);
  }

  if (guestPath === '/workspace/global' || guestPath.startsWith('/workspace/global/')) {
    const sub = guestPath.slice('/workspace/global'.length);
    return path.join(globalDir, sub);
  }

  if (guestPath.startsWith('/workspace/extra/')) {
    const extraPart = guestPath.slice('/workspace/extra/'.length); // e.g. "main-grocery/supermarket-list.md"
    for (const mount of group.containerConfig?.additionalMounts || []) {
      const mountName =
        mount.containerPath || path.basename(normalize(mount.hostPath));
      if (extraPart === mountName || extraPart.startsWith(mountName + '/')) {
        const sub = extraPart.slice(mountName.length); // leading "/" included or ""
        return normalize(path.join(normalize(mount.hostPath), sub));
      }
    }
  }

  return null;
}

// ─── IPC helpers ─────────────────────────────────────────────────────────────

function writeIpcMessage(
  groupFolder: string,
  chatJid: string,
  text: string,
): void {
  const dir = path.join(resolveGroupIpcPath(groupFolder), 'messages');
  fs.mkdirSync(dir, { recursive: true });
  const filename = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}.json`;
  const filepath = path.join(dir, filename);
  const tmp = `${filepath}.tmp`;
  fs.writeFileSync(
    tmp,
    JSON.stringify(
      {
        type: 'message',
        chatJid,
        text,
        groupFolder,
        timestamp: new Date().toISOString(),
      },
      null,
      2,
    ),
  );
  fs.renameSync(tmp, filepath);
}

function drainIpcInput(groupFolder: string): string[] {
  const inputDir = path.join(resolveGroupIpcPath(groupFolder), 'input');
  try {
    fs.mkdirSync(inputDir, { recursive: true });
    const files = fs
      .readdirSync(inputDir)
      .filter((f) => f.endsWith('.json'))
      .sort();

    const texts: string[] = [];
    for (const file of files) {
      const fp = path.join(inputDir, file);
      try {
        const data = JSON.parse(fs.readFileSync(fp, 'utf-8'));
        fs.unlinkSync(fp);
        if (data.type === 'message' && data.text) texts.push(data.text);
      } catch {
        try {
          fs.unlinkSync(fp);
        } catch { /* ignore */ }
      }
    }
    return texts;
  } catch {
    return [];
  }
}

function shouldClose(groupFolder: string): boolean {
  const sentinel = path.join(
    resolveGroupIpcPath(groupFolder),
    'input',
    '_close',
  );
  if (fs.existsSync(sentinel)) {
    try {
      fs.unlinkSync(sentinel);
    } catch { /* ignore */ }
    return true;
  }
  return false;
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

// ─── Ollama API ───────────────────────────────────────────────────────────────

async function callOllama(
  messages: OllamaMessage[],
  tools: OllamaTool[],
  signal: AbortSignal,
): Promise<OllamaMessage> {
  const body = {
    model: OLLAMA_MODEL,
    messages,
    tools,
    stream: false,
  };

  const res = await fetch(`${OLLAMA_URL}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  });

  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`Ollama ${res.status}: ${text.slice(0, 200)}`);
  }

  const data = (await res.json()) as { message: OllamaMessage };
  return data.message;
}

// ─── Tool executor ───────────────────────────────────────────────────────────

async function executeTool(
  name: string,
  args: Record<string, unknown>,
  group: RegisteredGroup,
  input: ContainerInput,
  onOutput: ((o: ContainerOutput) => Promise<void>) | undefined,
): Promise<string> {
  switch (name) {
    case 'read_file': {
      const guestPath = String(args.path ?? '');
      const hostPath = resolveGuestPath(guestPath, group);
      if (!hostPath) {
        return `Error: path not accessible: ${guestPath}`;
      }
      try {
        return fs.readFileSync(hostPath, 'utf-8');
      } catch (err) {
        return `Error reading ${guestPath}: ${err instanceof Error ? err.message : String(err)}`;
      }
    }

    case 'write_file': {
      const guestPath = String(args.path ?? '');
      const content = String(args.content ?? '');
      const hostPath = resolveGuestPath(guestPath, group);
      if (!hostPath) {
        return `Error: path not accessible: ${guestPath}`;
      }
      try {
        fs.mkdirSync(path.dirname(hostPath), { recursive: true });
        fs.writeFileSync(hostPath, content, 'utf-8');
        logger.debug({ hostPath }, 'Ollama agent wrote file');
        return 'File written successfully.';
      } catch (err) {
        return `Error writing ${guestPath}: ${err instanceof Error ? err.message : String(err)}`;
      }
    }

    case 'send_message': {
      const text = String(args.text ?? '');
      // Stream intermediate message to user immediately
      if (onOutput && text) {
        await onOutput({ status: 'success', result: text });
      }
      return 'Message sent.';
    }

    case 'notify_owner': {
      const text = String(args.text ?? '');
      const targets = input.allowedSendTargets ?? [];
      if (targets.length === 0) {
        return 'Error: no allowed send targets configured for this group.';
      }
      writeIpcMessage(group.folder, targets[0], text);
      logger.debug({ target: targets[0] }, 'Ollama agent notified owner');
      return 'Owner notified.';
    }

    default:
      return `Error: unknown tool "${name}"`;
  }
}

// ─── Tool loop ────────────────────────────────────────────────────────────────

/**
 * Run a single Ollama tool-calling loop given an existing messages array.
 * Appends to `messages` in place so conversation history is preserved across calls.
 * Returns the final assistant text (or null if the run was aborted).
 */
async function runToolLoop(
  messages: OllamaMessage[],
  tools: OllamaTool[],
  group: RegisteredGroup,
  input: ContainerInput,
  signal: AbortSignal,
  onOutput: ((o: ContainerOutput) => Promise<void>) | undefined,
): Promise<string | null> {
  const MAX_TOOL_ROUNDS = 20; // safety guard against infinite loops

  for (let round = 0; round < MAX_TOOL_ROUNDS; round++) {
    if (signal.aborted) return null;

    const response = await callOllama(messages, tools, signal);
    messages.push(response);

    if (!response.tool_calls || response.tool_calls.length === 0) {
      // Final text response
      return response.content || null;
    }

    // Execute each tool call and collect results
    for (const toolCall of response.tool_calls) {
      const toolName = toolCall.function.name;
      const toolArgs = toolCall.function.arguments;
      logger.debug({ tool: toolName, args: toolArgs }, 'Ollama tool call');

      const result = await executeTool(
        toolName,
        toolArgs,
        group,
        input,
        onOutput,
      );

      messages.push({
        role: 'tool',
        name: toolName,
        content: result,
      });
    }
    // Loop: send tool results back to the model
  }

  logger.warn({ group: group.name }, 'Ollama tool loop hit MAX_TOOL_ROUNDS');
  return null;
}

// ─── Public API ───────────────────────────────────────────────────────────────

const IPC_POLL_MS = 500;

/**
 * Run the Ollama agent — drop-in replacement for runContainerAgent.
 *
 * Instead of spawning a Docker container, this calls Ollama directly in-process,
 * translates container-style paths to host paths, and writes IPC files for
 * outbound messages (notify_owner) using the same format as the MCP server.
 */
export async function runOllamaAgent(
  group: RegisteredGroup,
  input: ContainerInput,
  onProcess: (proc: ChildProcess, containerName: string) => void,
  onOutput?: (output: ContainerOutput) => Promise<void>,
): Promise<ContainerOutput> {
  const abortController = new AbortController();

  // Provide a minimal fake ChildProcess so the queue's cancellation mechanism works.
  const fakeProc = new EventEmitter() as unknown as ChildProcess;
  (fakeProc as EventEmitter & { kill: () => void }).kill = () =>
    abortController.abort();
  const containerName = `ollama-${group.folder}-${Date.now()}`;
  onProcess(fakeProc, containerName);

  logger.info(
    { group: group.name, model: OLLAMA_MODEL },
    'Ollama agent starting',
  );

  // ── Build system prompt ────────────────────────────────────────────────────
  const groupDir = resolveGroupFolderPath(group.folder);
  const globalDir = path.join(GROUPS_DIR, 'global');

  let systemPrompt = '';

  // Prepend global CLAUDE.md (read-only context) for non-main groups
  if (!input.isMain) {
    const globalMd = path.join(globalDir, 'CLAUDE.md');
    if (fs.existsSync(globalMd)) {
      systemPrompt += fs.readFileSync(globalMd, 'utf-8') + '\n\n';
    }
  }

  const groupMd = path.join(groupDir, 'CLAUDE.md');
  if (fs.existsSync(groupMd)) {
    systemPrompt += fs.readFileSync(groupMd, 'utf-8');
  }

  if (!systemPrompt.trim()) {
    systemPrompt = 'You are a helpful assistant.';
  }

  // ── Clean up stale close sentinel ─────────────────────────────────────────
  const inputDir = path.join(resolveGroupIpcPath(group.folder), 'input');
  fs.mkdirSync(inputDir, { recursive: true });
  try {
    fs.unlinkSync(path.join(inputDir, '_close'));
  } catch { /* ignore */ }

  // ── Build initial prompt ───────────────────────────────────────────────────
  let firstPrompt = input.prompt;
  if (input.isScheduledTask) {
    firstPrompt = `[SCHEDULED TASK - sent automatically, not from the user]\n\n${firstPrompt}`;
  }
  // Drain any pending IPC input messages into the first prompt
  const pending = drainIpcInput(group.folder);
  if (pending.length > 0) {
    firstPrompt += '\n' + pending.join('\n');
  }

  const tools = buildTools(
    (input.allowedSendTargets ?? []).length > 0,
  );

  // Conversation history persisted across IPC follow-up turns within this session
  const messages: OllamaMessage[] = [
    { role: 'system', content: systemPrompt },
  ];

  try {
    // ── Query loop ─────────────────────────────────────────────────────────
    messages.push({ role: 'user', content: firstPrompt });

    const firstResult = await runToolLoop(
      messages,
      tools,
      group,
      input,
      abortController.signal,
      onOutput,
    );

    if (firstResult && onOutput) {
      await onOutput({ status: 'success', result: firstResult });
    }

    // Emit idle marker so the queue knows the agent is waiting
    if (onOutput) {
      await onOutput({ status: 'success', result: null });
    }

    // ── IPC follow-up loop ─────────────────────────────────────────────────
    while (!abortController.signal.aborted) {
      if (shouldClose(group.folder)) break;

      const followUps = drainIpcInput(group.folder);
      if (followUps.length > 0) {
        const userMsg = followUps.join('\n');
        messages.push({ role: 'user', content: userMsg });

        const result = await runToolLoop(
          messages,
          tools,
          group,
          input,
          abortController.signal,
          onOutput,
        );

        if (result && onOutput) {
          await onOutput({ status: 'success', result });
        }
        if (onOutput) {
          await onOutput({ status: 'success', result: null });
        }
      }

      await sleep(IPC_POLL_MS);
    }
  } catch (err) {
    if (abortController.signal.aborted) {
      return { status: 'success', result: null };
    }
    const error = err instanceof Error ? err.message : String(err);
    logger.error({ group: group.name, error }, 'Ollama agent error');
    if (onOutput) {
      await onOutput({ status: 'error', result: null, error });
    }
    return { status: 'error', result: null, error };
  }

  logger.info({ group: group.name }, 'Ollama agent done');
  return { status: 'success', result: null };
}
