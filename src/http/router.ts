import {
  createUnauthorizedMcpResponse,
  isAuthorizedConnectorPath,
  isAuthorizedRequest,
  timingSafeEqual
} from "./auth";
import { createApiPreflightResponse, withSecurityHeaders } from "./response";
import { getEmbedding as getCfEmbedding, getImageEmbedding } from "../embeddings";
import type { Env } from "../types";

interface AppRouteHandlers {
  processSubconscious(env: Env): Promise<void>;
  handleApiEntities(
    request: Request,
    env: Env,
    pathParts: string[]
  ): Promise<Response>;
  handleApiObservations(
    request: Request,
    env: Env,
    pathParts: string[]
  ): Promise<Response>;
  handleApiJournals(
    request: Request,
    env: Env,
    pathParts: string[]
  ): Promise<Response>;
  handleApiThreads(
    request: Request,
    env: Env,
    pathParts: string[]
  ): Promise<Response>;
  handleApiSearch(request: Request, env: Env): Promise<Response>;
  handleApiSurface(request: Request, env: Env): Promise<Response>;
  handleApiIdentity(
    request: Request,
    env: Env,
    pathParts: string[]
  ): Promise<Response>;
  handleApiRelations(
    request: Request,
    env: Env,
    pathParts: string[]
  ): Promise<Response>;
  handleApiImages(
    request: Request,
    env: Env,
    pathParts: string[]
  ): Promise<Response>;
  handleApiContext(
    request: Request,
    env: Env,
    pathParts: string[]
  ): Promise<Response>;
  handleApiBulkObservations(request: Request, env: Env): Promise<Response>;
  handleApiProcess(env: Env): Promise<Response>;
  handleApiOrient(env: Env): Promise<Response>;
  handleApiGround(env: Env): Promise<Response>;
  handleApiHealth(env: Env): Promise<Response>;
  handleApiHealthScores(env: Env): Promise<Response>;
  handleApiStats(env: Env): Promise<Response>;
  handleApiHeat(env: Env): Promise<Response>;
  handleApiRecent(env: Env): Promise<Response>;
  handleApiInnerWeather(env: Env): Promise<Response>;
  handleApiPatterns(env: Env): Promise<Response>;
  handleApiTensions(
    request: Request,
    env: Env,
    pathParts: string[]
  ): Promise<Response>;
  handleApiProposals(
    request: Request,
    env: Env,
    pathParts: string[]
  ): Promise<Response>;
  handleApiOrphans(
    request: Request,
    env: Env,
    pathParts: string[]
  ): Promise<Response>;
  handleApiArchive(
    request: Request,
    env: Env,
    pathParts: string[]
  ): Promise<Response>;
  handleApiObservationVersions(
    request: Request,
    env: Env,
    obsId: number
  ): Promise<Response>;
  handleMCPRequest(request: Request, env: Env): Promise<Response>;
}

/** Get the R2 path prefix from env or use default */
function r2Prefix(env: Env): string {
  return env.R2_PATH_PREFIX || "resonant-mind-images";
}

/** Get the worker's public URL */
function workerUrl(env: Env, request?: Request): string {
  if (env.WORKER_URL) return env.WORKER_URL.replace(/\/$/, "");
  if (request) return new URL(request.url).origin;
  return "https://localhost";
}

/** Get the signing secret (prefer dedicated secret, fall back to API key) */
function signingSecret(env: Env): string {
  return env.SIGNING_SECRET || env.MIND_API_KEY;
}

const ALLOWED_IMAGE_TYPES = new Set([
  "image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif", "image/svg+xml"
]);

/**
 * POST /api/images/upload — Direct file upload endpoint.
 * Accepts multipart/form-data. Bypasses MCP context window entirely.
 */
async function handleImageUpload(request: Request, env: Env): Promise<Response> {
  try {
    const formData = await request.formData();
    const file = formData.get("file") as File | null;
    const description = formData.get("description") as string || "";
    const entityName = formData.get("entity_name") as string || "";
    const emotion = formData.get("emotion") as string || "";
    const weight = formData.get("weight") as string || "medium";
    const context = formData.get("context") as string || "";
    const filename = formData.get("filename") as string || "";
    const observationId = formData.get("observation_id") as string || "";

    if (!file) return jsonResponse({ error: "No file provided" }, 400);
    if (!description) return jsonResponse({ error: "description is required" }, 400);
    if (file.size > 10 * 1024 * 1024) return jsonResponse({ error: "File too large. Max 10MB." }, 413);
    if (file.type && !ALLOWED_IMAGE_TYPES.has(file.type)) {
      return jsonResponse({ error: `Unsupported file type: ${file.type}. Allowed: png, jpeg, webp, gif, svg` }, 415);
    }

    const mimeType = file.type || "image/png";
    const rawBytes = new Uint8Array(await file.arrayBuffer());

    // Resolve entity
    let entityId: number | null = null;
    if (entityName) {
      const entity = await env.DB.prepare("SELECT id FROM entities WHERE name = ?").bind(entityName).first();
      if (entity) entityId = entity.id as number;
    }

    const prefix = r2Prefix(env);
    const date = new Date().toISOString().slice(0, 10).replace(/-/g, "");
    const safeName = (filename || description.slice(0, 50))
      .replace(/[^a-zA-Z0-9_-]/g, "_")
      .replace(/_+/g, "_")
      .slice(0, 60);
    const rawKey = `_tmp_${date}_${safeName}`;
    const webpKey = `${date}_${safeName}.webp`;

    await env.R2_IMAGES.put(rawKey, rawBytes, { httpMetadata: { contentType: mimeType } });

    let storedPath: string;
    let finalBytes: Uint8Array = rawBytes;
    let finalMime = mimeType;

    try {
      const baseUrl = workerUrl(env, request);
      const r2Url = `${baseUrl}/r2/${rawKey}`;
      const webpResponse = await fetch(r2Url, {
        cf: { image: { format: "webp", quality: 80, fit: "scale-down", width: 1920, height: 1920 } },
      });
      if (webpResponse.ok) {
        const webpBuffer = await webpResponse.arrayBuffer();
        finalBytes = new Uint8Array(webpBuffer);
        finalMime = "image/webp";
        await env.R2_IMAGES.put(webpKey, webpBuffer, { httpMetadata: { contentType: "image/webp" } });
        storedPath = `r2://${prefix}/${webpKey}`;
      } else {
        const ext = mimeType === "image/jpeg" ? ".jpg" : ".png";
        const fallbackKey = `${date}_${safeName}${ext}`;
        await env.R2_IMAGES.put(fallbackKey, rawBytes, { httpMetadata: { contentType: mimeType } });
        storedPath = `r2://${prefix}/${fallbackKey}`;
      }
    } catch {
      const ext = mimeType === "image/jpeg" ? ".jpg" : ".png";
      const fallbackKey = `${date}_${safeName}${ext}`;
      await env.R2_IMAGES.put(fallbackKey, rawBytes, { httpMetadata: { contentType: mimeType } });
      storedPath = `r2://${prefix}/${fallbackKey}`;
    }

    await env.R2_IMAGES.delete(rawKey).catch(() => {});

    // Insert into images table
    const result = await env.DB.prepare(`
      INSERT INTO images (path, description, context, emotion, weight, entity_id, observation_id)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `).bind(storedPath, description, context || null, emotion || null, weight, entityId, observationId ? parseInt(observationId) : null).run();

    const imageId = result.meta.last_row_id;

    // Generate multimodal embedding
    const contextText = [
      entityName ? `${entityName}:` : "", description,
      context ? `(${context})` : "", emotion ? `[${emotion}]` : ""
    ].filter(Boolean).join(" ");

    let embedded = false;
    let embeddingError: string | null = null;
    try {
      const embedding = await getImageEmbedding(env.AI, rawBytes.buffer as ArrayBuffer, mimeType, contextText);
      const metadata: Record<string, string> = {
        source: "image", description, weight, added_at: new Date().toISOString()
      };
      if (entityName) metadata.entity = entityName;
      if (context) metadata.context = context;
      if (emotion) metadata.emotion = emotion;
      metadata.path = storedPath;

      await env.VECTORS.upsert([{ id: `img-${imageId}`, values: embedding, metadata }]);
      embedded = true;
    } catch (e) {
      console.error("Multimodal embedding failed:", e);
      try {
        const textEmbedding = await getCfEmbedding(env.AI, contextText);
        const metadata: Record<string, string> = {
          source: "image", description, weight, added_at: new Date().toISOString()
        };
        if (entityName) metadata.entity = entityName;
        if (context) metadata.context = context;
        if (emotion) metadata.emotion = emotion;
        metadata.path = storedPath;
        await env.VECTORS.upsert([{ id: `img-${imageId}`, values: textEmbedding, metadata }]);
        embedded = true;
        embeddingError = "multimodal failed, used text fallback";
      } catch { /* text fallback also failed */ }
    }

    const originalSize = rawBytes.length;
    const finalSize = finalBytes.length;
    const saved = originalSize > finalSize ? Math.round((1 - finalSize / originalSize) * 100) : 0;

    return jsonResponse({
      id: imageId,
      path: storedPath,
      embedded,
      embedding_note: embeddingError,
      format: finalMime,
      original_size: `${Math.round(originalSize / 1024)}KB`,
      final_size: `${Math.round(finalSize / 1024)}KB`,
      compression: saved > 0 ? `${saved}% smaller` : "no conversion",
      entity: entityName || null,
      emotion: emotion || null,
    });
  } catch (e) {
    console.error("Upload error:", e);
    return jsonResponse({ error: "Image upload failed" }, 500);
  }
}

function jsonResponse(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json" }
  });
}

async function routeApiRequest(
  request: Request,
  env: Env,
  handlers: AppRouteHandlers,
  pathParts: string[]
): Promise<Response> {
  try {
    if (pathParts[1] === "entities") {
      return await handlers.handleApiEntities(request, env, pathParts);
    }

    if (pathParts[1] === "observations") {
      if (pathParts[2] === "bulk") {
        return await handlers.handleApiBulkObservations(request, env);
      }

      if (pathParts[3] === "versions") {
        return await handlers.handleApiObservationVersions(
          request,
          env,
          parseInt(pathParts[2], 10)
        );
      }

      return await handlers.handleApiObservations(request, env, pathParts);
    }

    if (pathParts[1] === "journals") {
      return await handlers.handleApiJournals(request, env, pathParts);
    }

    if (pathParts[1] === "threads") {
      return await handlers.handleApiThreads(request, env, pathParts);
    }

    if (pathParts[1] === "identity") {
      return await handlers.handleApiIdentity(request, env, pathParts);
    }

    if (pathParts[1] === "relations") {
      return await handlers.handleApiRelations(request, env, pathParts);
    }

    if (pathParts[1] === "images") {
      if (pathParts[2] === "upload" && request.method === "POST") {
        return await handleImageUpload(request, env);
      }
      return await handlers.handleApiImages(request, env, pathParts);
    }

    if (pathParts[1] === "context") {
      return await handlers.handleApiContext(request, env, pathParts);
    }

    if (pathParts[1] === "search") return await handlers.handleApiSearch(request, env);
    if (pathParts[1] === "surface") return await handlers.handleApiSurface(request, env);
    if (pathParts[1] === "orient") return await handlers.handleApiOrient(env);
    if (pathParts[1] === "ground") return await handlers.handleApiGround(env);
    if (pathParts[1] === "health") return await handlers.handleApiHealth(env);
    if (pathParts[1] === "health-scores") return await handlers.handleApiHealthScores(env);
    if (pathParts[1] === "stats") return await handlers.handleApiStats(env);
    if (pathParts[1] === "heat") return await handlers.handleApiHeat(env);
    if (pathParts[1] === "recent") return await handlers.handleApiRecent(env);
    if (pathParts[1] === "inner-weather") return await handlers.handleApiInnerWeather(env);
    if (pathParts[1] === "patterns") return await handlers.handleApiPatterns(env);

    if (pathParts[1] === "process" && request.method === "POST") {
      return await handlers.handleApiProcess(env);
    }

    if (pathParts[1] === "tensions") {
      return await handlers.handleApiTensions(request, env, pathParts);
    }

    if (pathParts[1] === "proposals") {
      return await handlers.handleApiProposals(request, env, pathParts);
    }

    if (pathParts[1] === "orphans") {
      return await handlers.handleApiOrphans(request, env, pathParts);
    }

    if (pathParts[1] === "archive") {
      return await handlers.handleApiArchive(request, env, pathParts);
    }

    return jsonResponse({ error: "Unknown API endpoint" }, 404);
  } catch (error) {
    console.error("API error:", error);
    return jsonResponse({ error: "Internal server error" }, 500);
  }
}

export async function routeRequest(
  request: Request,
  env: Env,
  handlers: AppRouteHandlers
): Promise<Response> {
  const url = new URL(request.url);
  const prefix = r2Prefix(env);

  if (url.pathname === "/health") {
    return withSecurityHeaders(
      jsonResponse({ status: "ok", service: "resonant-mind" }),
      request,
      env
    );
  }

  // Image viewing: /img/{id} with signed URL (no API key exposed)
  if (url.pathname.startsWith("/img/") && env.R2_IMAGES) {
    const expires = url.searchParams.get("expires");
    const sig = url.searchParams.get("sig");
    const imageId = url.pathname.slice(5);

    if (!expires || !sig) return new Response("Missing signature", { status: 401 });
    if (parseInt(expires) < Math.floor(Date.now() / 1000)) return new Response("URL expired", { status: 403 });

    // Verify HMAC with timing-safe comparison
    const encoder = new TextEncoder();
    const secret = signingSecret(env);
    const key = await crypto.subtle.importKey("raw", encoder.encode(secret), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
    const signature = await crypto.subtle.sign("HMAC", key, encoder.encode(`${imageId}:${expires}`));
    const expectedSig = Array.from(new Uint8Array(signature)).map(b => b.toString(16).padStart(2, "0")).join("");
    if (!timingSafeEqual(sig, expectedSig)) return new Response("Invalid signature", { status: 401 });

    const img = await env.DB.prepare("SELECT path FROM images WHERE id = ?").bind(imageId).first();
    if (!img?.path || !String(img.path).startsWith(`r2://${prefix}/`)) {
      return new Response("Not found", { status: 404 });
    }
    const r2Key = String(img.path).replace(`r2://${prefix}/`, "");
    const object = await env.R2_IMAGES.get(r2Key);
    if (!object) return new Response("Not found", { status: 404 });
    return new Response(object.body, {
      headers: {
        "Content-Type": object.httpMetadata?.contentType || "image/webp",
        "Cache-Control": "private, max-age=3600",
      }
    });
  }

  // Internal R2 serving (used by cf.image transform for WebP conversion)
  // All R2 access requires authentication — no _tmp_ bypass
  if (url.pathname.startsWith("/r2/") && env.R2_IMAGES) {
    if (!isAuthorizedRequest(request, env)) {
      return new Response("Unauthorized", { status: 401 });
    }
    const key = url.pathname.slice(4);
    const object = await env.R2_IMAGES.get(key);
    if (!object) return new Response("Not found", { status: 404 });
    return new Response(object.body, {
      headers: { "Content-Type": object.httpMetadata?.contentType || "image/png" }
    });
  }

  if (url.pathname.startsWith("/api/")) {
    if (request.method === "OPTIONS") {
      return createApiPreflightResponse(request, env);
    }

    if (!isAuthorizedRequest(request, env)) {
      return withSecurityHeaders(
        jsonResponse({ error: "Unauthorized" }, 401),
        request,
        env,
        { api: true }
      );
    }

    const pathParts = url.pathname.split("/").filter(Boolean);
    return withSecurityHeaders(
      await routeApiRequest(request, env, handlers, pathParts),
      request,
      env,
      { api: true }
    );
  }

  if (url.pathname === "/process" && request.method === "POST") {
    if (!isAuthorizedRequest(request, env)) {
      return withSecurityHeaders(
        jsonResponse({ error: "Unauthorized" }, 401),
        request,
        env,
        { api: true }
      );
    }

    await handlers.processSubconscious(env);
    return withSecurityHeaders(
      jsonResponse({ status: "processed" }),
      request,
      env,
      { api: true }
    );
  }

  if (url.pathname === "/subconscious") {
    if (!isAuthorizedRequest(request, env)) {
      return withSecurityHeaders(
        jsonResponse({ error: "Unauthorized" }, 401),
        request,
        env,
        { api: true }
      );
    }

    const result = await env.DB.prepare(
      "SELECT data FROM subconscious WHERE state_type = 'daemon' ORDER BY updated_at DESC LIMIT 1"
    ).first();

    return withSecurityHeaders(
      jsonResponse(result?.data ? JSON.parse(result.data as string) : {}),
      request,
      env,
      { api: true }
    );
  }

  const usesConnectorPath = isAuthorizedConnectorPath(url, env);
  if ((url.pathname === "/mcp" || usesConnectorPath) && request.method === "POST") {
    if (!usesConnectorPath && !isAuthorizedRequest(request, env)) {
      return withSecurityHeaders(
        createUnauthorizedMcpResponse(),
        request,
        env
      );
    }

    return withSecurityHeaders(
      await handlers.handleMCPRequest(request, env),
      request,
      env
    );
  }

  return withSecurityHeaders(
    new Response("Resonant Mind", {
      headers: { "Content-Type": "text/plain" }
    }),
    request,
    env
  );
}
