/**
 * Thought-signature relay for Google's OpenAI-compatibility layer.
 *
 * Gemini 3.x attaches an opaque `extra_content.google.thought_signature` to
 * every function-call turn and REFUSES (HTTP 400) any follow-up request whose
 * history contains a function call without one. Standard OpenAI clients don't
 * know that field, drop it, and then break on their second request — a
 * protocol violation introduced by Google, so the proxy heals it here:
 *
 * 1. Response side: harvest signatures from chat-completion responses
 *    (non-streaming JSON and streaming SSE) keyed by tool-call id.
 * 2. Request side: for assistant messages whose tool calls carry no
 *    signature, inject the cached one (Google emits one signature per turn,
 *    attached to the first call; a message-level signature validates the
 *    whole turn). On a cache miss the turn is degraded to plain text, which
 *    Gemini accepts without any signature — so a request can never fail
 *    because of a missing signature, no matter the cache state.
 */

/** Signature lookup/write-through backend (KV-backed in production, memory otherwise). */
export interface SignatureStore {
    get(id: string): Promise<string | undefined>;
    put(id: string, signature: string): Promise<void>;
}

/** Tool-call entry in an OpenAI chat completion (request or response shape). */
interface ToolCallLike {
    id?: unknown;
    extra_content?: unknown;
}

/** Read `extra_content.google.thought_signature` without trusting the shape. */
export function signatureOf(call: unknown): string | undefined {
    if (typeof call !== "object" || call === null) return undefined;
    const extra = (call as ToolCallLike).extra_content;
    if (typeof extra !== "object" || extra === null) return undefined;
    const google = (extra as Record<string, unknown>).google;
    if (typeof google !== "object" || google === null) return undefined;
    const signature = (google as Record<string, unknown>).thought_signature;
    return typeof signature === "string" && signature.length > 0
        ? signature
        : undefined;
}

/** Collect (toolCallId, signature) pairs from any object tree. */
export function harvestPairs(root: unknown): [string, string][] {
    const pairs: [string, string][] = [];
    const visit = (node: unknown): void => {
        if (Array.isArray(node)) {
            for (const item of node) visit(item);
            return;
        }
        if (typeof node !== "object" || node === null) return;
        const record = node as Record<string, unknown>;
        const id = record.id;
        const signature = signatureOf(record);
        if (typeof id === "string" && id.length > 0 && signature) {
            pairs.push([id, signature]);
        }
        for (const value of Object.values(record)) visit(value);
    };
    visit(root);
    return pairs;
}

// ---------------------------------------------------------------------------
// Request side
// ---------------------------------------------------------------------------

export interface BackfillResult {
    /** True when messages were rewritten (signature injected or turn degraded). */
    changed: boolean;
    /** Number of assistant turns degraded to plain text (cache misses). */
    degraded: number;
    /** Number of assistant turns healed from cache. */
    healed: number;
}

/**
 * Heal conversation history in place so no function call travels without its
 * thought signature. Cache hits inject a message-level signature; misses
 * degrade the whole assistant turn (calls + following tool results) to text.
 * @param messages - the parsed request's `messages` array (mutated in place).
 */
export async function backfillThoughtSignatures(
    messages: unknown,
    store: SignatureStore,
): Promise<BackfillResult> {
    const result: BackfillResult = { changed: false, degraded: 0, healed: 0 };
    if (!Array.isArray(messages)) return result;

    for (let i = 0; i < messages.length; i++) {
        const message = messages[i];
        if (typeof message !== "object" || message === null) continue;
        const record = message as Record<string, unknown>;
        if (record.role !== "assistant") continue;
        const calls = record.tool_calls;
        if (!Array.isArray(calls) || calls.length === 0) continue;
        if (signatureOf(record)) continue; // correct client — nothing to do

        // Google emits one signature per assistant turn (on the first call),
        // and a message-level signature validates the whole turn.
        let signature: string | undefined;
        for (const call of calls) {
            const id =
                typeof call === "object" && call !== null
                    ? (call as ToolCallLike).id
                    : undefined;
            if (typeof id !== "string" || id.length === 0) continue;
            signature = await store.get(id);
            if (signature) break;
        }

        if (signature) {
            record.extra_content = { google: { thought_signature: signature } };
            result.changed = true;
            result.healed++;
        } else {
            i = degradeTurn(messages, i);
            result.changed = true;
            result.degraded++;
        }
    }
    return result;
}

/**
 * Rewrite an assistant tool-call turn (messages[index]) into plain text and
 * fold its tool results into user messages. Returns the index of the last
 * message consumed, so the caller's loop resumes after the rewritten block.
 */
function degradeTurn(messages: unknown[], index: number): number {
    const record = messages[index] as Record<string, unknown>;
    const calls = (record.tool_calls ?? []) as unknown[];
    const parts: string[] = [textOf(record.content)];
    for (const call of calls) {
        const fn =
            typeof call === "object" && call !== null
                ? ((call as Record<string, unknown>).function as
                      | Record<string, unknown>
                      | undefined)
                : undefined;
        const name = typeof fn?.name === "string" ? fn.name : "tool";
        const args = typeof fn?.arguments === "string" ? fn.arguments : "{}";
        parts.push(`[${name} called with ${args}]`);
    }
    const text = parts.filter(Boolean).join("\n");
    messages[index] = { role: "assistant", content: text };

    let last = index;
    for (let j = index + 1; j < messages.length; j++) {
        const next = messages[j];
        if (typeof next !== "object" || next === null) break;
        const tool = next as Record<string, unknown>;
        if (tool.role !== "tool") break;
        const callId =
            typeof tool.tool_call_id === "string" ? tool.tool_call_id : "";
        const fnName = extractName(calls, callId);
        const body = textOf(tool.content) || "{}";
        messages[j] = {
            role: "user",
            content: `[tool result for ${fnName}${callId ? ` (${callId})` : ""}]\n${body}`,
        };
        last = j;
    }
    return last;
}

function extractName(calls: unknown[], callId: string): string {
    for (const call of calls) {
        if (
            typeof call === "object" &&
            call !== null &&
            (call as ToolCallLike).id === callId
        ) {
            const fn = (call as Record<string, unknown>).function;
            if (
                typeof fn === "object" &&
                fn !== null &&
                typeof (fn as Record<string, unknown>).name === "string"
            ) {
                return (fn as Record<string, unknown>).name as string;
            }
        }
    }
    return "tool";
}

/** Render OpenAI content (string or content-part array) as plain text. */
function textOf(content: unknown): string {
    if (typeof content === "string") return content;
    if (Array.isArray(content)) {
        return content
            .map((part) =>
                typeof part === "object" &&
                part !== null &&
                typeof (part as Record<string, unknown>).text === "string"
                    ? ((part as Record<string, unknown>).text as string)
                    : "",
            )
            .filter(Boolean)
            .join("\n");
    }
    return "";
}

// ---------------------------------------------------------------------------
// Response side — streaming
// ---------------------------------------------------------------------------

/**
 * Incremental SSE watcher: feed every `data:` payload of a chat-completions
 * stream; signatures found on tool-call deltas are harvested into the store.
 * Fire-and-forget: put() rejections are swallowed (the cache is best-effort;
 * a miss only costs quality through degradation, never correctness).
 */
export function createStreamWatcher(store: SignatureStore): {
    observe: (payload: unknown) => void;
    /** Await all pending cache writes (for waitUntil / end-of-request). */
    done: () => Promise<void>;
} {
    const writes: Promise<void>[] = [];
    let lastCallId: string | undefined;
    return {
        observe: (payload) => {
            const pairs = harvestPairs(payload);
            for (const [id, signature] of pairs) {
                lastCallId = id;
                writes.push(store.put(id, signature).catch(() => {}));
            }
            // Delta-level signature without a tool-call id in the same chunk:
            // attribute it to the most recent call of this stream (Google
            // attaches the turn's signature to the first call, but chunk
            // layouts may vary between models).
            if (pairs.length === 0 && lastCallId) {
                const choices = (payload as { choices?: unknown })?.choices;
                const delta = Array.isArray(choices)
                    ? (choices[0] as { delta?: unknown } | undefined)?.delta
                    : undefined;
                const deltaLevel = signatureOf(delta);
                if (deltaLevel) {
                    writes.push(
                        store.put(lastCallId, deltaLevel).catch(() => {}),
                    );
                }
            }
        },
        done: () => Promise.all(writes).then(() => {}),
    };
}

/**
 * Pump a tee'd response branch through the watcher without disturbing the
 * client-facing branch. Never rejects.
 */
export async function pumpSse(
    stream: ReadableStream<Uint8Array>,
    watcher: ReturnType<typeof createStreamWatcher>,
): Promise<void> {
    const reader = stream.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    try {
        for (;;) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            let newline: number;
            while ((newline = buffer.indexOf("\n")) >= 0) {
                const line = buffer.slice(0, newline).trim();
                buffer = buffer.slice(newline + 1);
                if (!line.startsWith("data:")) continue;
                const payload = line.slice(5).trim();
                if (!payload || payload === "[DONE]") continue;
                try {
                    watcher.observe(JSON.parse(payload));
                } catch {
                    // not JSON — not a chat chunk, ignore
                }
            }
        }
    } catch {
        // client disconnected or upstream aborted — cache whatever we got
    } finally {
        reader.releaseLock();
        try {
            await stream.cancel();
        } catch {
            // already closed
        }
    }
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

const MEMORY_LIMIT = 512;
const KV_TTL_SECONDS = 24 * 60 * 60;
const KV_PREFIX = "tsig:";

/**
 * Write-through signature cache: process memory first (hits within the same
 * isolate are instant), KV namespace for durability across isolates/PoPs.
 * Every KV failure is swallowed — a miss only triggers text degradation.
 */
export class SignatureCache implements SignatureStore {
    private memory = new Map<string, string>();
    private kv: KVNamespace | undefined;

    constructor(kv: KVNamespace | undefined) {
        this.kv = kv;
    }

    async get(id: string): Promise<string | undefined> {
        const cached = this.memory.get(id);
        if (cached !== undefined) return cached;
        if (!this.kv) return undefined;
        try {
            const signature = await this.kv.get(KV_PREFIX + id);
            if (signature) {
                this.remember(id, signature);
                return signature;
            }
        } catch {
            // KV unavailable — degradation covers it
        }
        return undefined;
    }

    async put(id: string, signature: string): Promise<void> {
        this.remember(id, signature);
        if (!this.kv) return;
        try {
            await this.kv.put(KV_PREFIX + id, signature, {
                expirationTtl: KV_TTL_SECONDS,
            });
        } catch {
            // KV unavailable — this isolate's memory still covers hot paths
        }
    }

    private remember(id: string, signature: string): void {
        if (this.memory.has(id)) this.memory.delete(id);
        this.memory.set(id, signature);
        if (this.memory.size > MEMORY_LIMIT) {
            const oldest = this.memory.keys().next().value;
            if (oldest !== undefined) this.memory.delete(oldest);
        }
    }
}
