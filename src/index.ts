import { Hono } from "hono";
import { cors } from "hono/cors";
import { parseConfig, updateConfig } from "./config";
import { getStorage } from "./storage";
import type { Credential } from "./types";
import { hashKey, validateClientKey } from "./utils";
import { getAccessToken, rewritePathForVertexAI } from "./vertexai";

const app = new Hono<{ Bindings: CloudflareBindings }>();

// Enable CORS for all routes
app.use("/*", cors());

// Update configuration
app.post("/_config", async (c) => {
    const secret = c.env.CLIENT_KEY_VALIDATION_SECRET;
    if (!secret) {
        return c.json({ error: "Update not allowed" }, 403);
    }

    const authHeader = c.req.header("Authorization");
    if (!authHeader || authHeader !== `Bearer ${secret}`) {
        return c.json({ error: "Unauthorized" }, 401);
    }

    const storage = getStorage(c.env);
    if (storage.readonly) {
        return c.json({ error: "Storage is read-only" }, 403);
    }

    try {
        const data = await c.req.json();
        await updateConfig(c.env, data);
        return c.json({ success: true });
    } catch (error) {
        return c.json({ error: (error as Error).message }, 500);
    }
});

// Get configuration
app.get("/_config", async (c) => {
    const secret = c.env.CLIENT_KEY_VALIDATION_SECRET;
    if (!secret) {
        return c.json({ error: "Read not allowed" }, 403);
    }

    const authHeader = c.req.header("Authorization");
    if (!authHeader || authHeader !== `Bearer ${secret}`) {
        return c.json({ error: "Unauthorized" }, 401);
    }

    try {
        const config = await parseConfig(c.env);
        const storage = getStorage(c.env);

        // Convert Config to ConfigData format for response
        const keys = config.keys.map((k) =>
            typeof k.key === "string" ? k.key : JSON.stringify(k.key),
        );
        const baseUrls = config.keys.map((k) => k.baseUrl);
        return c.json({ keys, baseUrls, readonly: storage.readonly });
    } catch (error) {
        return c.json({ error: (error as Error).message }, 500);
    }
});

// Get authorization header for a key
async function getAuthHeader(
    key: string | Credential,
): Promise<[key: string, value: string]> {
    if (typeof key === "string") {
        return ["x-goog-api-key", key];
    } else {
        const token = await getAccessToken(key);
        return ["authorization", `Bearer ${token}`];
    }
}

// Proxy all requests to Gemini API
app.all("/*", async (c) => {
    const config = await parseConfig(c.env);

    if (config.keys.length === 0) {
        return c.json({ error: "No API keys configured" }, 500);
    }

    let path = c.req.path;
    const url = new URL(c.req.url);
    const headers = new Headers(c.req.raw.headers);
    // Client key: x-goog-api-key first, then fall back to Authorization Bearer
    // (the OpenAI SDK sends its api_key as a Bearer token)
    let clientKey = headers.get("x-goog-api-key");
    if (!clientKey) {
        const authorization = headers.get("authorization");
        if (authorization?.startsWith("Bearer ")) {
            clientKey = authorization.slice(7).trim();
        }
    }
    if (!clientKey) {
        return c.json({ error: "Missing API key" }, 400);
    }

    if (c.env.CLIENT_KEY_VALIDATION_SECRET) {
        const payload = await validateClientKey(
            clientKey,
            c.env.CLIENT_KEY_VALIDATION_SECRET,
        );
        if (!payload) {
            return c.json({ error: "Invalid API key" }, 403);
        }

        const allowedEndpoints: string[] = payload.allowed_endpoints || [".*"];
        const isAllowed = allowedEndpoints.some((pattern) =>
            new RegExp(pattern).test(path),
        );

        if (!isAllowed) {
            return c.json({ error: "Endpoint not allowed" }, 403);
        }
    }

    // Google's OpenAI-compat layer rejects unknown/unsupported fields with a
    // 400 instead of ignoring them. Common OpenAI clients (e.g. SillyTavern)
    // always send several of those, so strip the known-bad ones upfront.
    let openaiBody: Record<string, unknown> | null = null;
    let openaiBodyStr: string | null = null;
    let openaiRawBody: ArrayBuffer | null = null;
    if (path.includes("/openai/") && c.req.method === "POST") {
        openaiRawBody = await c.req.arrayBuffer();
        try {
            const parsed = JSON.parse(
                new TextDecoder().decode(openaiRawBody),
            ) as Record<string, unknown>;
            for (const field of [
                "frequency_penalty",
                "presence_penalty",
                "logit_bias",
                "seed",
                "logprobs",
                "top_logprobs",
            ]) {
                delete parsed[field];
            }
            if (Array.isArray(parsed.stop) && parsed.stop.length === 0) {
                delete parsed.stop; // empty stop arrays are rejected upstream
            }
            openaiBody = parsed;
            openaiBodyStr = JSON.stringify(parsed);
        } catch {
            // Not JSON — forward the original bytes untouched
        }
    }

    // Try each key until one succeeds
    let lastError: Error | null = null;

    const keyConfigs = config.keys.sort(() => Math.random() - 0.5); // Shuffle keys
    for (const keyConfig of keyConfigs) {
        try {
            // Build target URL with this key's base URL
            if (typeof keyConfig.key === "object") {
                path = rewritePathForVertexAI(
                    path,
                    keyConfig.key.project_id,
                    "global",
                );
            }
            const targetUrl = new URL("." + path, keyConfig.baseUrl);
            targetUrl.search = url.search;

            const authHeader = await getAuthHeader(keyConfig.key);

            // Forward the request
            headers.delete("host");
            headers.delete("x-goog-api-key");
            headers.set(
                "cf-aig-metadata",
                JSON.stringify({
                    clientKey,
                    serverKeyHash: await hashKey(keyConfig.key),
                }),
            );
            headers.set(authHeader[0], authHeader[1]);
            if (typeof keyConfig.key === "string") {
                if (path.includes("/openai/")) {
                    // The OpenAI compatibility layer only accepts Bearer auth
                    headers.set("authorization", `Bearer ${keyConfig.key}`);
                } else {
                    // Native endpoints: strip the client JWT before forwarding
                    headers.delete("authorization");
                }
            }

            // Google names the offending field in 400 responses ("Unknown
            // name \"x\": Cannot find field.") — drop it and retry the same
            // key so exotic clients still work without a field blacklist.
            for (let attempt = 0; ; attempt++) {
                const response = await fetch(targetUrl.toString(), {
                    method: c.req.method,
                    headers,
                    body:
                        c.req.method !== "GET" && c.req.method !== "HEAD"
                            ? (openaiBodyStr ?? openaiRawBody ?? c.req.raw.body)
                            : undefined,
                });

                if (
                    response.status === 400 &&
                    openaiBody !== null &&
                    attempt < 8
                ) {
                    const probe = await response
                        .clone()
                        .text()
                        .catch(() => "");
                    const match = probe.match(/Unknown name "([^"]+)":/);
                    if (match && match[1] in openaiBody) {
                        delete openaiBody[match[1]];
                        openaiBodyStr = JSON.stringify(openaiBody);
                        continue;
                    }
                    // Some models reject penalties with a semantic error
                    // instead of an unknown-field error — drop them and retry.
                    if (/Penalty is not enabled/i.test(probe)) {
                        let removed = false;
                        for (const field of [
                            "presence_penalty",
                            "frequency_penalty",
                        ]) {
                            if (field in openaiBody) {
                                delete openaiBody[field];
                                removed = true;
                            }
                        }
                        if (removed) {
                            openaiBodyStr = JSON.stringify(openaiBody);
                            continue;
                        }
                    }
                }

                // If successful, return the response
                if (response.ok || response.status < 500) {
                    // Return response with same headers
                    const responseHeaders = new Headers(response.headers);
                    responseHeaders.set("Access-Control-Allow-Origin", "*");

                    return new Response(response.body, {
                        status: response.status,
                        statusText: response.statusText,
                        headers: responseHeaders,
                    });
                }

                lastError = new Error(
                    `HTTP ${response.status}: ${response.statusText}`,
                );
                break;
            }
        } catch (error) {
            lastError = error as Error;
            continue;
        }
    }

    return c.json(
        {
            error: "All API keys failed",
            message: lastError?.message || "Unknown error",
        },
        500,
    );
});

export default app;
