/**
 * Unit tests for the thought-signature relay.
 * Run: pnpm test  (node --experimental-strip-types)
 */
import assert from "node:assert/strict";
import { test } from "node:test";
import {
    backfillThoughtSignatures,
    createStreamWatcher,
    harvestPairs,
    pumpSse,
    SignatureCache,
    signatureOf,
} from "../src/thought-signature.ts";

class MemoryStore {
    map = new Map<string, string>();
    async get(id: string) {
        return this.map.get(id);
    }
    async put(id: string, signature: string) {
        this.map.set(id, signature);
    }
}

const SIG = "sig-" + "A".repeat(64);

test("signatureOf reads the nested field and rejects junk", () => {
    assert.equal(
        signatureOf({ extra_content: { google: { thought_signature: SIG } } }),
        SIG,
    );
    assert.equal(signatureOf({ extra_content: { google: {} } }), undefined);
    assert.equal(
        signatureOf({ extra_content: { google: { thought_signature: "" } } }),
        undefined,
    );
    assert.equal(signatureOf(null), undefined);
    assert.equal(signatureOf("x"), undefined);
});

test("harvestPairs finds id+signature pairs in a response tree", () => {
    const body = {
        choices: [
            {
                message: {
                    role: "assistant",
                    tool_calls: [
                        {
                            id: "call_1",
                            type: "function",
                            function: { name: "f" },
                            extra_content: {
                                google: { thought_signature: SIG },
                            },
                        },
                        {
                            id: "call_2",
                            type: "function",
                            function: { name: "g" },
                        },
                    ],
                    extra_content: { google: { thought_signature: SIG } },
                },
            },
        ],
        usage: { total_tokens: 10 },
    };
    assert.deepEqual(harvestPairs(body), [["call_1", SIG]]);
    assert.deepEqual(harvestPairs({ nope: 1 }), []);
});

test("backfill leaves correct clients untouched", async () => {
    const store = new MemoryStore();
    const messages = [
        { role: "user", content: "hi" },
        {
            role: "assistant",
            tool_calls: [
                {
                    id: "call_1",
                    type: "function",
                    function: { name: "f", arguments: "{}" },
                },
            ],
            extra_content: { google: { thought_signature: SIG } },
        },
        { role: "tool", tool_call_id: "call_1", content: "ok" },
    ];
    const before = JSON.stringify(messages);
    const result = await backfillThoughtSignatures(messages, store);
    assert.deepEqual(result, { changed: false, degraded: 0, healed: 0 });
    assert.equal(JSON.stringify(messages), before);
});

test("backfill injects a cached signature at message level", async () => {
    const store = new MemoryStore();
    await store.put("call_9", SIG);
    const messages = [
        { role: "user", content: "hi" },
        {
            role: "assistant",
            tool_calls: [
                {
                    id: "call_x",
                    type: "function",
                    function: { name: "f", arguments: "{}" },
                },
                {
                    id: "call_9",
                    type: "function",
                    function: { name: "g", arguments: "{}" },
                },
            ],
        },
        { role: "tool", tool_call_id: "call_x", content: "ok" },
    ];
    const result = await backfillThoughtSignatures(messages, store);
    assert.equal(result.changed, true);
    assert.equal(result.healed, 1);
    assert.equal(result.degraded, 0);
    assert.deepEqual((messages[1] as any).extra_content, {
        google: { thought_signature: SIG },
    });
    assert.equal((messages[1] as any).tool_calls.length, 2);
});

test("backfill degrades unknown turns to plain text, tool results become user", async () => {
    const store = new MemoryStore();
    const messages = [
        { role: "system", content: "sys" },
        { role: "user", content: "weather?" },
        {
            role: "assistant",
            content: "Let me check.",
            tool_calls: [
                {
                    id: "call_unknown",
                    type: "function",
                    function: {
                        name: "get_weather",
                        arguments: '{"city":"Tokyo"}',
                    },
                },
            ],
        },
        { role: "tool", tool_call_id: "call_unknown", content: "22C sunny" },
        { role: "user", content: "thanks" },
    ];
    const result = await backfillThoughtSignatures(messages, store);
    assert.equal(result.changed, true);
    assert.equal(result.degraded, 1);
    assert.equal(result.healed, 0);

    const degraded = messages[2] as any;
    assert.equal(degraded.role, "assistant");
    assert.equal(degraded.tool_calls, undefined);
    assert.equal(degraded.extra_content, undefined);
    assert.match(degraded.content, /Let me check\./);
    assert.match(
        degraded.content,
        /\[get_weather called with \{"city":"Tokyo"\}\]/,
    );

    const folded = messages[3] as any;
    assert.equal(folded.role, "user");
    assert.match(
        folded.content,
        /\[tool result for get_weather \(call_unknown\)\]/,
    );
    assert.match(folded.content, /22C sunny/);

    assert.equal((messages[4] as any).role, "user"); // untouched
});

test("stream watcher harvests tool-call signatures from SSE payloads", async () => {
    const store = new MemoryStore();
    const watcher = createStreamWatcher(store);
    watcher.observe({ choices: [{ delta: { role: "assistant" } }] });
    watcher.observe({
        choices: [
            {
                delta: {
                    tool_calls: [
                        {
                            id: "call_s1",
                            type: "function",
                            function: { name: "f" },
                            extra_content: {
                                google: { thought_signature: SIG },
                            },
                        },
                    ],
                },
            },
        ],
    });
    watcher.observe({
        choices: [
            {
                delta: {
                    tool_calls: [
                        {
                            id: "call_s2",
                            type: "function",
                            function: { name: "g" },
                        },
                    ],
                },
            },
        ],
        finish_reason: "stop",
    });
    await watcher.done();
    assert.equal(await store.get("call_s1"), SIG);
    assert.equal(await store.get("call_s2"), undefined);
});

test("pumpSse parses a byte stream end to end", async () => {
    const store = new MemoryStore();
    const watcher = createStreamWatcher(store);
    const sse = [
        'data: {"choices":[{"delta":{"role":"assistant"}}]}',
        "",
        `data: {"choices":[{"delta":{"tool_calls":[{"id":"call_p","function":{"name":"f"},"extra_content":{"google":{"thought_signature":"${SIG}"}}}]}}]}`,
        "",
        "data: [DONE]",
        "",
    ].join("\n");
    const stream = new ReadableStream({
        start(controller) {
            controller.enqueue(new TextEncoder().encode(sse));
            controller.close();
        },
    });
    await pumpSse(stream, watcher);
    await watcher.done();
    assert.equal(await store.get("call_p"), SIG);
});

test("SignatureCache falls back between memory and KV", async () => {
    const backing = new Map<string, string>();
    const fakeKv = {
        get: async (key: string) => backing.get(key),
        put: async (key: string, value: string) => backing.set(key, value),
    } as unknown as KVNamespace;

    const cache = new SignatureCache(fakeKv);
    await cache.put("call_kv", SIG);
    assert.equal(backing.get("tsig:call_kv"), SIG); // written through

    const fresh = new SignatureCache(fakeKv); // cold memory, warm KV
    assert.equal(await fresh.get("call_kv"), SIG);

    const memoryOnly = new SignatureCache(undefined);
    await memoryOnly.put("call_mem", SIG);
    assert.equal(await memoryOnly.get("call_mem"), SIG);
    assert.equal(await memoryOnly.get("call_missing"), undefined);
});
