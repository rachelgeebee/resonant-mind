export interface Env {
  DB: D1Database;
  HYPERDRIVE: Hyperdrive;
  VECTORS: VectorizeIndex;
  AI: Ai;
  R2_IMAGES: R2Bucket;
  GEMINI_API_KEY?: string;
  MIND_API_KEY: string;
  SIGNING_SECRET?: string;
  MCP_CONNECTOR_SECRET?: string;
  WEATHER_API_KEY?: string;
  DASHBOARD_ALLOWED_ORIGIN?: string;
  INTERNAL_KEY?: string;
  WORKER_URL?: string;
  R2_PATH_PREFIX?: string;
  LOCATION_NAME?: string;
  LOCATION_LAT?: string;
  LOCATION_LON?: string;
  LOCATION_TZ?: string;
}

export interface MCPRequest {
  jsonrpc: "2.0";
  id?: string | number | null;
  method: string;
  params?: Record<string, unknown>;
}

export interface MCPResponse {
  jsonrpc: "2.0";
  id: string | number | null;
  result?: unknown;
  error?: { code: number; message: string };
}

export interface MCPToolDefinition {
  name: string;
  description: string;
  inputSchema: Record<string, unknown>;
}

export type MCPToolHandler = (
  env: Env,
  params: Record<string, unknown>
) => Promise<string>;

export type MCPToolHandlerMap = Record<string, MCPToolHandler>;
