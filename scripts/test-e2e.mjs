/**
 * End-to-end test: mock Google upstream + real wrangler dev.
 *
 * The mock reproduces the two upstream behaviors this proxy heals:
 *   - error bodies wrapped in an array ([{"error":{...}}])
 *   - 400 on any history whose function calls lack a thought signature
 *
 * Run: pnpm test:e2e
 */
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { rm, writeFile } from "node:fs/promises";
import { createServer } from "node:http";

const MOCK_PORT = 9599;
const PROXY_PORT = 8799;
const PROXY = `http://127.0.0.1:${PROXY_PORT}`;
const SIG = "e2e-" + "S".repeat(64);

// ---------------------------------------------------------------------------
// mock upstream
// ---------------------------------------------------------------------------
let lastForwarded = null;

function hasUnsignedToolCall(messages) {
    for (const m of messages ?? []) {
        if (m.role !== "assistant" || !Array.isArray(m.tool_calls)) continue;
        if (!m.extra_content?.google?.thought_signature) return true;
    }
    return false;
}

function toolCallResponse(id) {
    return {
        choices: [
            {
                finish_reason: "tool_calls",
                index: 0,
                message: {
                    role: "assistant",
                    tool_calls: [
                        {
                            id,
                            type: "function",
                            function: {
                                name: "get_weather",
                                arguments: '{"city":"Tokyo"}',
                            },
                            extra_content: {
                                google: { thought_signature: SIG },
                            },
                        },
                    ],
                },
            },
        ],
        usage: { completion_tokens: 1, prompt_tokens: 5, total_tokens: 6 },
    };
}

function sendJson(res, status, body) {
    res.writeHead(status, { "Content-Type": "application/json" });
    res.end(JSON.stringify(body));
}

const mockServer = createServer((req, res) => {
    const url = new URL(req.url, "http://localhost");
    if (url.pathname === "/_last") {
        return sendJson(res, 200, lastForwarded);
    }
    if (url.searchParams.get("error") === "1") {
        return sendJson(res, 400, [
            {
                error: {
                    code: 400,
                    message: "Mock upstream failure",
                    status: "INVALID_ARGUMENT",
                },
            },
        ]);
    }
    if (req.method !== "POST" || !url.pathname.endsWith("/chat/completions")) {
        return sendJson(res, 200, { object: "list", data: [] });
    }
    let raw = "";
    req.on("data", (chunk) => (raw += chunk));
    req.on("end", () => {
        const body = JSON.parse(raw);
        lastForwarded = body;
        // Mirrors Google's "Unknown name" rejection, wrapped in an array like
        // the real upstream so the retry probe is exercised end to end.
        if (body.store !== undefined) {
            return sendJson(res, 400, [
                {
                    error: {
                        code: 400,
                        message:
                            'Invalid JSON payload received. Unknown name "store": Cannot find field.',
                        status: "INVALID_ARGUMENT",
                        details: [
                            {
                                "@type":
                                    "type.googleapis.com/google.rpc.BadRequest",
                                fieldViolations: [
                                    {
                                        description:
                                            'Invalid JSON payload received. Unknown name "store": Cannot find field.',
                                    },
                                ],
                            },
                        ],
                    },
                },
            ]);
        }
        if (hasUnsignedToolCall(body.messages)) {
            return sendJson(res, 400, [
                {
                    error: {
                        code: 400,
                        message:
                            "Function call is missing a thought_signature in functionCall parts.",
                        status: "INVALID_ARGUMENT",
                    },
                },
            ]);
        }
        if (body.stream) {
            const chunks = [
                toolCallResponse("call_stream_1"),
                toolCallResponse("call_stream_2"),
            ].map(
                (r) =>
                    `data: ${JSON.stringify({ ...r, object: "chat.completion.chunk" })}\n\n`,
            );
            chunks.push("data: [DONE]\n\n");
            res.writeHead(200, { "Content-Type": "text/event-stream" });
            return res.end(chunks.join(""));
        }
        sendJson(
            res,
            200,
            toolCallResponse(`call_${Math.random().toString(36).slice(2, 10)}`),
        );
    });
});
mockServer.on("error", (err) => {
    console.error("mock upstream error:", err);
    process.exit(1);
});

// ---------------------------------------------------------------------------
// wrangler dev
// ---------------------------------------------------------------------------
await writeFile(
    ".dev.vars",
    `GEMINI_API_KEY=mock-key\nGEMINI_API_BASE_URL=http://127.0.0.1:${MOCK_PORT}\n`,
);
const wrangler = spawn(
    "npx",
    ["wrangler", "dev", "--port", String(PROXY_PORT)],
    {
        shell: true,
        stdio: "pipe",
        env: { ...process.env, CI: "1", WRANGLER_SEND_METRICS: "false" },
    },
);
wrangler.stdout.on("data", (d) => process.stdout.write(`[wrangler] ${d}`));
wrangler.stderr.on("data", (d) => process.stderr.write(`[wrangler] ${d}`));

/** Kill the whole process tree (shell:true leaves grandchildren on Windows). */
function killWrangler() {
    if (wrangler.pid && wrangler.exitCode === null) {
        spawn("taskkill", ["/F", "/T", "/PID", String(wrangler.pid)], {
            shell: true,
        });
    }
}
process.on("exit", killWrangler);
process.on("SIGINT", () => process.exit(1));
process.on("unhandledRejection", (err) => {
    console.error("unhandled rejection:", err);
    process.exit(1);
});
process.on("uncaughtException", (err) => {
    console.error("uncaught exception:", err);
    process.exit(1);
});

async function waitForProxy(timeoutMs = 90_000) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        try {
            await fetch(PROXY + "/");
            return;
        } catch {
            await new Promise((r) => setTimeout(r, 1000));
        }
    }
    throw new Error("wrangler dev did not become ready");
}

async function chat(body) {
    const res = await fetch(PROXY + "/v1beta/openai/chat/completions", {
        method: "POST",
        headers: {
            Authorization: "Bearer test",
            "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
    });
    return { status: res.status, text: await res.text() };
}

try {
    await new Promise((resolve, reject) => {
        mockServer.once("error", reject);
        mockServer.listen(MOCK_PORT, "127.0.0.1", () => {
            mockServer.off("error", reject);
            resolve();
        });
    });
    console.log(`mock upstream ready on :${MOCK_PORT}`);
    await waitForProxy();
    console.log(`wrangler dev ready on :${PROXY_PORT}`);

    // T1: array-wrapped upstream errors are unwrapped to the standard object
    const t1 = await fetch(PROXY + "/v1beta/openai/chat/completions?error=1", {
        method: "POST",
        headers: {
            Authorization: "Bearer test",
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            model: "m",
            messages: [{ role: "user", content: "hi" }],
        }),
    });
    const t1body = JSON.parse(await t1.text());
    assert.equal(t1.status, 400);
    assert.equal(
        t1body.error?.code,
        400,
        "error body must be the unwrapped OBJECT",
    );
    console.log("T1 pass: array-wrapped error normalized to object");

    // T2: agent loop round 1 (non-stream) — harvest signature by tool-call id
    const t2a = await chat({
        model: "m",
        messages: [{ role: "user", content: "weather?" }],
    });
    assert.equal(t2a.status, 200);
    const callId = JSON.parse(t2a.text).choices[0].message.tool_calls[0].id;

    // T2 round 2: same id, no signature — proxy must heal from cache
    const t2b = await chat({
        model: "m",
        messages: [
            { role: "user", content: "weather?" },
            {
                role: "assistant",
                tool_calls: [
                    {
                        id: callId,
                        type: "function",
                        function: {
                            name: "get_weather",
                            arguments: '{"city":"Tokyo"}',
                        },
                    },
                ],
            },
            { role: "tool", tool_call_id: callId, content: "22C" },
        ],
    });
    assert.equal(
        t2b.status,
        200,
        `expected healed 200, got ${t2b.status}: ${t2b.text.slice(0, 200)}`,
    );
    assert.equal(
        lastForwarded.messages[1].extra_content?.google?.thought_signature,
        SIG,
        "message-level signature must be injected",
    );
    console.log("T2 pass: non-stream signature harvested and re-injected");

    // T3: streaming round 1 — bypass parser must harvest while client streams
    const t3res = await fetch(PROXY + "/v1beta/openai/chat/completions", {
        method: "POST",
        headers: {
            Authorization: "Bearer test",
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            model: "m",
            stream: true,
            messages: [{ role: "user", content: "weather?" }],
        }),
    });
    assert.equal(t3res.headers.get("content-type") ?? "", "text/event-stream");
    const t3text = await t3res.text();
    assert.match(t3text, /call_stream_1/);
    assert.match(t3text, /\[DONE\]/);
    await new Promise((r) => setTimeout(r, 1500)); // let the bypass finish

    const t3b = await chat({
        model: "m",
        messages: [
            { role: "user", content: "weather?" },
            {
                role: "assistant",
                tool_calls: [
                    {
                        id: "call_stream_1",
                        type: "function",
                        function: {
                            name: "get_weather",
                            arguments: '{"city":"Tokyo"}',
                        },
                    },
                ],
            },
            { role: "tool", tool_call_id: "call_stream_1", content: "22C" },
        ],
    });
    assert.equal(
        t3b.status,
        200,
        `expected stream-harvested heal, got ${t3b.status}: ${t3b.text.slice(0, 200)}`,
    );
    console.log("T3 pass: SSE bypass harvested signature; follow-up healed");

    // T4: unknown id — turn must degrade to text instead of failing
    const t4 = await chat({
        model: "m",
        messages: [
            { role: "user", content: "weather?" },
            {
                role: "assistant",
                tool_calls: [
                    {
                        id: "call_never_seen",
                        type: "function",
                        function: {
                            name: "get_weather",
                            arguments: '{"city":"Tokyo"}',
                        },
                    },
                ],
            },
            { role: "tool", tool_call_id: "call_never_seen", content: "22C" },
        ],
    });
    assert.equal(
        t4.status,
        200,
        `expected degraded 200, got ${t4.status}: ${t4.text.slice(0, 200)}`,
    );
    const degradedAssistant = lastForwarded.messages[1];
    assert.equal(
        degradedAssistant.tool_calls,
        undefined,
        "tool_calls must be stripped on degradation",
    );
    assert.match(degradedAssistant.content, /get_weather called with/);
    assert.equal(
        lastForwarded.messages[2].role,
        "user",
        "tool result folded into a user message",
    );
    console.log("T4 pass: unknown signature degrades to plain text");

    // T5: correct clients pass through untouched (signature already present)
    const t5 = await chat({
        model: "m",
        messages: [
            { role: "user", content: "weather?" },
            {
                role: "assistant",
                tool_calls: [
                    {
                        id: "call_ok",
                        type: "function",
                        function: { name: "get_weather", arguments: "{}" },
                    },
                ],
                extra_content: { google: { thought_signature: SIG } },
            },
            { role: "tool", tool_call_id: "call_ok", content: "22C" },
        ],
    });
    assert.equal(t5.status, 200);
    assert.equal(
        lastForwarded.messages[1].extra_content.google.thought_signature,
        SIG,
    );
    assert.equal(lastForwarded.messages[1].tool_calls.length, 1);
    console.log("T5 pass: signed history passes through untouched");

    // T6: Google-style "Unknown name" errors strip the offending field and
    // retry — `store` was historically rejected and the JSON-escaping on the
    // wire broke the old regex entirely.
    const t6 = await chat({
        model: "m",
        messages: [{ role: "user", content: "hi" }],
        store: false,
    });
    assert.equal(
        t6.status,
        200,
        `expected retry 200, got ${t6.status}: ${t6.text.slice(0, 200)}`,
    );
    assert.equal(
        "store" in lastForwarded,
        false,
        `store must be stripped, lastForwarded keys: ${Object.keys(lastForwarded).join(",")}`,
    );
    console.log("T6 pass: unknown-field retry strips store and succeeds");

    console.log("\nALL E2E TESTS PASSED");
} finally {
    killWrangler();
    await new Promise((r) => mockServer.close(r));
    await rm(".dev.vars", { force: true });
}
