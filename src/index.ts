/**
 * Resonant Mind - Persistent Memory MCP Server
 * Cognitive infrastructure for AI systems — semantic memory, emotional processing, identity continuity
 */

import { routeRequest } from "./http/router";
import { handleMcpProtocolRequest } from "./mcp/protocol";
import { createD1Adapter } from "./adapter";
import { createVectorAdapter } from "./vectors";
import { getEmbedding as getGeminiEmbedding, getImageEmbedding, generateText as geminiGenerateText } from "./embeddings";
import type {
  Env,
  MCPToolDefinition,
  MCPToolHandlerMap
} from "./types";

const RESONANT_MIND_VERSION = "3.1.1";

// Surface pool configuration
const SURFACE_POOL_RATIOS = { core: 0.5, novelty: 0.2, dormant: 0.2, edge: 0.1 };
const VECTOR_SCORE_CORE = 0.75;  // Gemini Embedding 2 scores ~15pts higher than BGE
const VECTOR_SCORE_EDGE = 0.55;
const NOVELTY_FLOORS = { heavy: 0.3, medium: 0.2, light: 0.1 };
const NOVELTY_DECAY_RATES = { heavy: 0.08, medium: 0.12, light: 0.15 };
const NOVELTY_TIME_RECOVERY_RATE = 0.01; // per day since last surfaced
const NOVELTY_TIME_RECOVERY_CAP = 0.3;
const ORPHAN_AGE_DAYS = 30;
const ARCHIVE_AGE_DAYS = 30;

// Multi-factor retrieval scoring (Phase 1)
const SEARCH_SCORING = { alpha: 0.50, beta: 0.20, gamma: 0.20, delta: 0.10 };
const RECENCY_DECAY_RATE = 0.02;  // exp(-rate * days), half-life ~35 days
const ACCESS_GROWTH_RATE = 0.1;   // 1 - exp(-rate * count), saturates ~20 accesses

// Contradiction detection thresholds (Phase 2)
const CONTRADICTION_SIMILARITY_THRESHOLD = 0.80;
const AUTO_SUPERSEDE_THRESHOLD = 0.85;

// Consolidation and reflection (Phase 3)
const CONSOLIDATION_MIN_OBS = 10;
const CONSOLIDATION_MAX_ENTITIES_PER_RUN = 3;
const REFLECTION_MIN_OBS = 5;
const ACCESS_DECAY_PENALTY = 0.05;
const ACCESS_DECAY_AGE_DAYS = 30;

// Normalize text to ASCII - prevents Unicode homoglyph issues
function normalizeText(text: string | null | undefined): string | null {
  if (!text) return null;
  return text.normalize('NFKD').replace(/[^a-zA-Z0-9\s,.\-]/g, '').trim() || null;
}

// Tool definitions for MCP
const TOOLS: MCPToolDefinition[] = [
  {
    name: "mind_orient",
    description: "First call on wake - get identity anchor, current context, relational state",
    inputSchema: { type: "object", properties: {}, required: [] }
  },
  {
    name: "mind_ground",
    description: "Second call on wake - get active threads, recent work, recent journals",
    inputSchema: { type: "object", properties: {}, required: [] }
  },
  {
    name: "mind_thread",
    description: "Manage threads (intentions across sessions)",
    inputSchema: {
      type: "object",
      properties: {
        action: { type: "string", enum: ["list", "add", "resolve", "update", "delete"] },
        status: { type: "string" },
        content: { type: "string" },
        thread_type: { type: "string" },
        context: { type: "string" },
        priority: { type: "string" },
        thread_id: { type: "string" },
        resolution: { type: "string" },
        new_content: { type: "string" },
        new_priority: { type: "string" },
        new_status: { type: "string" },
        add_note: { type: "string" }
      },
      required: ["action"]
    }
  },
  {
    name: "mind_write",
    description: "Write to cognitive databases (entity, observation, relation, journal, image)",
    inputSchema: {
      type: "object",
      properties: {
        type: { type: "string", enum: ["entity", "observation", "relation", "journal"] },
        name: { type: "string" },
        entity_type: { type: "string" },
        entity_name: { type: "string" },
        observations: { type: "array", items: { type: "string" } },
        context: { type: "string" },
        salience: { type: "string" },
        emotion: { type: "string" },
        weight: { type: "string", enum: ["light", "medium", "heavy"], description: "Emotional weight for observations/images" },
        certainty: { type: "string", enum: ["tentative", "believed", "known"], description: "How certain: tentative=exploring, believed=accept it, known=verified fact" },
        source: { type: "string", enum: ["conversation", "realization", "external", "inferred"], description: "Origin: conversation=discussed, realization=insight, external=told, inferred=concluded" },
        from_entity: { type: "string" },
        to_entity: { type: "string" },
        relation_type: { type: "string" },
        entry: { type: "string" },
        tags: { type: "array", items: { type: "string" } },
        path: { type: "string", description: "For images: file path or URL" },
        description: { type: "string", description: "For images: what the image shows" },
        observation_id: { type: "number", description: "For images: link to a specific observation" }
      },
      required: ["type"]
    }
  },
  {
    name: "mind_search",
    description: "Search memories using semantic similarity",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string" },
        context: { type: "string" },
        n_results: { type: "number" },
        keyword: { type: "string", description: "Filter results to only those containing this keyword/phrase (case-insensitive)" },
        source: { type: "string", description: "Filter by source (e.g., 'gpt_recovery', 'conversation', 'realization')" },
        entity: { type: "string", description: "Filter by entity name" },
        weight: { type: "string", enum: ["light", "medium", "heavy"], description: "Filter by emotional weight" },
        date_from: { type: "string", description: "Filter by source_date >= YYYY-MM-DD" },
        date_to: { type: "string", description: "Filter by source_date <= YYYY-MM-DD" },
        type: { type: "string", enum: ["observation", "entity", "journal", "image"], description: "Filter by memory type" },
        include_expired: { type: "boolean", description: "Include superseded/expired observations (default: false)" }
      },
      required: ["query"]
    }
  },

  {
    name: "mind_feel_toward",
    description: "Track, check, or clear relational state toward someone",
    inputSchema: {
      type: "object",
      properties: {
        person: { type: "string" },
        feeling: { type: "string" },
        intensity: { type: "string", enum: ["whisper", "present", "strong", "overwhelming"] },
        clear: { type: "boolean", description: "Clear all relational state for this person" },
        clear_id: { type: "number", description: "Delete a specific relational state entry by ID" }
      },
      required: ["person"]
    }
  },
  {
    name: "mind_identity",
    description: "Read or write identity graph",
    inputSchema: {
      type: "object",
      properties: {
        action: { type: "string", enum: ["read", "write", "delete"] },
        section: { type: "string" },
        content: { type: "string" },
        weight: { type: "number" },
        connections: { type: "string" }
      }
    }
  },
  {
    name: "mind_context",
    description: "Current context layer - situational awareness",
    inputSchema: {
      type: "object",
      properties: {
        action: { type: "string", enum: ["read", "set", "update", "clear"] },
        scope: { type: "string" },
        content: { type: "string" },
        links: { type: "string" },
        id: { type: "string" }
      }
    }
  },
  {
    name: "mind_health",
    description: "Check cognitive health stats",
    inputSchema: { type: "object", properties: {}, required: [] }
  },
  {
    name: "mind_list_entities",
    description: "List all entities, optionally filtered by type or context",
    inputSchema: {
      type: "object",
      properties: {
        entity_type: { type: "string", description: "Filter by type (person, concept, project, etc.)" },
        context: { type: "string", description: "Filter by context (default, relational-models, etc.)" },
        limit: { type: "number", description: "Max results (default 50)" }
      },
      required: []
    }
  },
  {
    name: "mind_read_entity",
    description: "Read an entity with all its observations and relations",
    inputSchema: {
      type: "object",
      properties: {
        name: { type: "string", description: "Entity name to read" },
        context: { type: "string", description: "Context to search in (optional, searches all if not specified)" }
      },
      required: ["name"]
    }
  },
  {
    name: "mind_sit",
    description: "Sit with an emotional observation - engage with it, add a note about what arises. Increments sit count and may shift charge level.",
    inputSchema: {
      type: "object",
      properties: {
        observation_id: { type: "number", description: "ID of the observation to sit with" },
        text_match: { type: "string", description: "Or find by text content (partial match)" },
        query: { type: "string", description: "Or find by semantic search (closest meaning match)" },
        sit_note: { type: "string", description: "What arose while sitting with this" }
      },
      required: ["sit_note"]
    }
  },
  {
    name: "mind_resolve",
    description: "Mark an emotional observation as metabolized - link it to a resolution or insight that processed it",
    inputSchema: {
      type: "object",
      properties: {
        observation_id: { type: "number", description: "ID of the observation to resolve" },
        text_match: { type: "string", description: "Or find by text content (partial match)" },
        resolution_note: { type: "string", description: "How this was resolved/metabolized" },
        linked_observation_id: { type: "number", description: "Optional: ID of another observation that provided the resolution" }
      },
      required: ["resolution_note"]
    }
  },
  {
    name: "mind_surface",
    description: "Surface observations - resonant (emotional/mood-based, default) or spark (random associative thinking)",
    inputSchema: {
      type: "object",
      properties: {
        mode: { type: "string", enum: ["resonant", "spark"], description: "resonant (default) = mood/emotion based 3-pool surfacing. spark = random associative with hot-entity bias." },
        query: { type: "string", description: "Optional association trigger - a word, feeling, or concept to surface around" },
        include_metabolized: { type: "boolean", description: "Also show resolved observations (default false)" },
        limit: { type: "number", description: "Max results (default 10 for resonant, 5 for spark)" },
        weight_bias: { type: "string", enum: ["light", "medium", "heavy"], description: "Spark mode: bias toward this weight" }
      },
      required: []
    }
  },
  {
    name: "mind_edit",
    description: "Edit an existing observation, image, or journal",
    inputSchema: {
      type: "object",
      properties: {
        observation_id: { type: "number", description: "ID of observation to edit" },
        image_id: { type: "number", description: "ID of image to edit" },
        journal_id: { type: "number", description: "ID of journal to edit" },
        text_match: { type: "string", description: "Find observation by content (partial match)" },
        description_match: { type: "string", description: "Find image by description (partial match)" },
        new_content: { type: "string", description: "New content for observation/journal (or new description for image)" },
        new_weight: { type: "string", enum: ["light", "medium", "heavy"], description: "New weight" },
        new_emotion: { type: "string", description: "New emotion tag" },
        new_context: { type: "string", description: "New context (images only)" },
        new_path: { type: "string", description: "New path (images only)" }
      },
      required: []
    }
  },
  {
    name: "mind_delete",
    description: "Delete any memory: observation, entity, journal, relation, image, thread, or tension",
    inputSchema: {
      type: "object",
      properties: {
        observation_id: { type: "number", description: "ID of observation to delete" },
        entity_name: { type: "string", description: "Name of entity to delete (cascades observations)" },
        text_match: { type: "string", description: "Find observation by text (partial match)" },
        journal_id: { type: "number", description: "ID of journal to delete" },
        relation_id: { type: "number", description: "ID of relation to delete" },
        image_id: { type: "number", description: "ID of image to delete (removes from R2 + embedding)" },
        thread_id: { type: "string", description: "ID of thread to delete" },
        tension_id: { type: "string", description: "ID of tension to delete" }
      },
      required: []
    }
  },
  {
    name: "mind_entity",
    description: "Manage entities - set salience, edit properties, merge duplicates, bulk archive",
    inputSchema: {
      type: "object",
      properties: {
        action: {
          type: "string",
          enum: ["set_salience", "edit", "merge", "archive_old"],
          description: "Action to perform"
        },
        entity_id: { type: "number", description: "Entity ID to modify" },
        entity_name: { type: "string", description: "Entity name (alternative to ID)" },
        context: { type: "string", description: "Context for entity lookup" },
        salience: {
          type: "string",
          enum: ["foundational", "active", "background", "archive"],
          description: "New salience level (for set_salience)"
        },
        new_name: { type: "string", description: "New name (for edit)" },
        new_type: { type: "string", description: "New entity type (for edit)" },
        new_context: { type: "string", description: "New context (for edit)" },
        merge_into_id: { type: "number", description: "Target entity ID to merge into (for merge)" },
        merge_from_id: { type: "number", description: "Source entity ID to merge from and delete (for merge)" },
        older_than_days: { type: "number", description: "Archive entities older than X days (for archive_old)" },
        entity_type_filter: { type: "string", description: "Only archive this entity type (for archive_old)" }
      },
      required: ["action"]
    }
  },
  {
    name: "mind_consolidate",
    description: "Review and consolidate recent observations - find patterns, merge duplicates",
    inputSchema: {
      type: "object",
      properties: {
        days: { type: "number", description: "How many days back to look (default 7)" },
        context: { type: "string", description: "Limit to specific context" }
      },
      required: []
    }
  },
  {
    name: "mind_read",
    description: "Read entities/observations from a database",
    inputSchema: {
      type: "object",
      properties: {
        scope: { type: "string", enum: ["all", "context", "recent", "observation"], description: "all, context, recent, or observation (by ID)" },
        context: { type: "string", description: "Which database (for scope='context')" },
        hours: { type: "number", description: "How far back (for scope='recent')" },
        observation_id: { type: "number", description: "Observation ID (for scope='observation')" }
      },
      required: ["scope"]
    }
  },
  {
    name: "mind_timeline",
    description: "Trace a topic through time - semantic search ordered chronologically",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string" },
        start_date: { type: "string", description: "Optional start (YYYY-MM-DD)" },
        end_date: { type: "string", description: "Optional end (YYYY-MM-DD)" },
        n_results: { type: "number", description: "Max results" }
      },
      required: ["query"]
    }
  },
  {
    name: "mind_patterns",
    description: "Analyze recurring patterns - what's alive, what's surfacing",
    inputSchema: {
      type: "object",
      properties: {
        days: { type: "number", description: "How many days back to analyze (default 7)" },
        include_all_time: { type: "boolean", description: "Include foundational patterns" }
      },
      required: []
    }
  },
  {
    name: "mind_inner_weather",
    description: "Check current inner weather - what's coloring experience right now",
    inputSchema: {
      type: "object",
      properties: {},
      required: []
    }
  },
  {
    name: "mind_tension",
    description: "Tension space - hold productive contradictions that simmer",
    inputSchema: {
      type: "object",
      properties: {
        action: { type: "string", enum: ["list", "add", "sit", "resolve", "delete"] },
        pole_a: { type: "string", description: "One side of the tension" },
        pole_b: { type: "string", description: "The other side" },
        context: { type: "string", description: "Why this tension matters" },
        tension_id: { type: "string", description: "For sit/resolve actions" },
        resolution: { type: "string", description: "How it resolved (for resolve action)" }
      },
      required: ["action"]
    }
  },
  {
    name: "mind_proposals",
    description: "Review and act on daemon-proposed connections from co-surfacing patterns",
    inputSchema: {
      type: "object",
      properties: {
        action: {
          type: "string",
          enum: ["list", "accept", "reject"],
          description: "list shows pending proposals, accept creates relation, reject dismisses"
        },
        proposal_id: {
          type: "number",
          description: "Required for accept/reject actions"
        },
        relation_type: {
          type: "string",
          description: "For accept - what kind of relation to create (e.g., 'connects_to', 'resonates_with')"
        }
      },
      required: []
    }
  },
  {
    name: "mind_orphans",
    description: "Review and rescue orphaned observations that haven't surfaced",
    inputSchema: {
      type: "object",
      properties: {
        action: {
          type: "string",
          enum: ["list", "surface", "archive"],
          description: "list shows orphans, surface forces one to appear, archive removes from tracking"
        },
        observation_id: {
          type: "number",
          description: "Required for surface/archive actions"
        }
      },
      required: []
    }
  },
  {
    name: "mind_archive",
    description: "Explore and manage the deep archive - memories that have faded but aren't forgotten",
    inputSchema: {
      type: "object",
      properties: {
        action: {
          type: "string",
          enum: ["list", "rescue", "explore"],
          description: "list shows archived memories, rescue brings back to active, explore searches the deep"
        },
        observation_id: {
          type: "number",
          description: "For rescue action - bring this observation back to active memory"
        },
        query: {
          type: "string",
          description: "For explore action - semantic search within archived memories only"
        }
      },
      required: []
    }
  },
  {
    name: "mind_store_image",
    description: "Store, view, or search visual memories. For store: pass file_path (a PreToolUse hook uploads the file locally, bypassing context limits). Handles WebP conversion, R2 upload, and multimodal Gemini embedding.",
    inputSchema: {
      type: "object",
      properties: {
        action: { type: "string", enum: ["store", "view", "search", "delete"], description: "store=upload new image, view=browse images, search=semantic image search, delete=remove image" },
        file_path: { type: "string", description: "For store: local file path (hook uploads directly, bypasses context)" },
        image_data: { type: "string", description: "For store: base64-encoded image data (small images only)" },
        mime_type: { type: "string", description: "For store: image/png or image/jpeg" },
        filename: { type: "string", description: "For store: meaningful filename (will be sanitized)" },
        description: { type: "string", description: "For store: what the image shows" },
        entity_name: { type: "string", description: "For store/view: linked entity name" },
        emotion: { type: "string", description: "For store/view: emotional tone" },
        weight: { type: "string", enum: ["light", "medium", "heavy"], description: "For store/view: significance" },
        context: { type: "string", description: "For store: when/why created" },
        observation_id: { type: "number", description: "For store: link to a specific observation" },
        query: { type: "string", description: "For search: semantic search text" },
        random: { type: "boolean", description: "For view: random selection" },
        limit: { type: "number", description: "For view/search: max results (default 5)" }
      },
      required: ["action"]
    }
  }
];

// Helper: R2 path prefix (configurable via env)
function r2Prefix(env: Env): string {
  return env.R2_PATH_PREFIX || "resonant-mind-images";
}

// Helper: Worker URL for self-referencing (image signed URLs, cf.image transform)
function workerUrl(env: Env): string {
  if (env.WORKER_URL) return env.WORKER_URL.replace(/\/$/, "");
  return "https://localhost";
}

// Generate embedding using Gemini Embedding 2
async function getEmbedding(env: Env, text: string): Promise<number[]> {
  return getGeminiEmbedding(env.GEMINI_API_KEY, text);
}

async function searchVectors(env: Env, query: string, topK: number) {
  const embedding = await getEmbedding(env, query);
  return env.VECTORS.query(embedding, { topK, returnMetadata: "all" });
}

// Generate unique ID
function generateId(prefix: string): string {
  const now = new Date();
  const timestamp = now.toISOString().replace(/[-:T.Z]/g, '').slice(0, 14);
  const random = Math.random().toString(36).slice(2, 6);
  return `${prefix}-${timestamp}-${random}`;
}

// Generate signed image URL (1 hour expiry, no API key exposed)
async function imageUrl(imageId: number | string, env: Env): Promise<string> {
  const expires = Math.floor(Date.now() / 1000) + 3600; // 1 hour
  const encoder = new TextEncoder();
  const secret = env.SIGNING_SECRET || env.MIND_API_KEY;
  const key = await crypto.subtle.importKey("raw", encoder.encode(secret), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  const signature = await crypto.subtle.sign("HMAC", key, encoder.encode(`${imageId}:${expires}`));
  const sig = Array.from(new Uint8Array(signature)).map(b => b.toString(16).padStart(2, "0")).join("");
  return `${workerUrl(env)}/img/${imageId}?expires=${expires}&sig=${sig}`;
}

// ============================================================
// WEATHER & TIME CONTEXT - Inner weather infrastructure
// ============================================================

// Location config (configurable via env vars, defaults to UTC)
function getLocation(env: Env) {
  return {
    name: env.LOCATION_NAME || "Unknown",
    latitude: parseFloat(env.LOCATION_LAT || "0"),
    longitude: parseFloat(env.LOCATION_LON || "0"),
    timezone: env.LOCATION_TZ || "UTC"
  };
}

// Weather mood mappings - textures to draw from
const WEATHER_MOODS: Record<string, {energy: string; textures: string[]}> = {
  "clear": {energy: "bright", textures: ["clear-headed", "expansive", "energized"]},
  "cloudy": {energy: "muted", textures: ["contemplative", "soft", "introspective"]},
  "rainy": {energy: "inward", textures: ["reflective", "tender", "creative"]},
  "stormy": {energy: "intense", textures: ["restless", "raw", "electric"]},
  "snowy": {energy: "still", textures: ["hushed", "peaceful", "magical"]},
  "foggy": {energy: "liminal", textures: ["dreamy", "uncertain", "between-worlds"]}
};

// Weather code to atmosphere mapping (Open-Meteo codes)
const WEATHER_CODES: Record<number, string> = {
  0: "clear", 1: "clear", 2: "cloudy", 3: "cloudy",
  45: "foggy", 48: "foggy",
  51: "rainy", 53: "rainy", 55: "rainy", 61: "rainy", 63: "rainy", 65: "rainy",
  66: "rainy", 67: "rainy", 80: "rainy", 81: "rainy",
  71: "snowy", 73: "snowy", 75: "snowy", 77: "snowy", 85: "snowy", 86: "snowy",
  82: "stormy", 95: "stormy", 96: "stormy", 99: "stormy"
};

interface WeatherData {
  atmosphere: string;
  temp_f: number | null;
  location: string;
  error?: string;
  weather_code?: number;
}

async function getCurrentWeather(env?: Env): Promise<WeatherData> {
  const loc = env ? getLocation(env) : { name: "Unknown", latitude: 0, longitude: 0, timezone: "UTC" };
  try {
    const apiKey = env?.WEATHER_API_KEY;
    if (!apiKey) {
      return {atmosphere: "clear", temp_f: null, location: loc.name};
    }
    const q = loc.latitude && loc.longitude ? `${loc.latitude},${loc.longitude}` : loc.name;
    const url = `https://api.weatherapi.com/v1/current.json?key=${apiKey}&q=${encodeURIComponent(q)}`;

    const response = await fetch(url);
    if (!response.ok) {
      return {atmosphere: "clear", temp_f: null, location: loc.name, error: `API status ${response.status}`};
    }

    const data = await response.json() as any;
    const conditionText = (data?.current?.condition?.text || "").toLowerCase();
    const temp = data?.current?.temp_f;

    let atmosphere = "clear";
    if (conditionText.includes("thunder") || conditionText.includes("storm")) {
      atmosphere = "stormy";
    } else if (conditionText.includes("snow") || conditionText.includes("sleet") || conditionText.includes("ice")) {
      atmosphere = "snowy";
    } else if (conditionText.includes("rain") || conditionText.includes("drizzle") || conditionText.includes("shower")) {
      atmosphere = "rainy";
    } else if (conditionText.includes("fog") || conditionText.includes("mist")) {
      atmosphere = "foggy";
    } else if (conditionText.includes("cloud") || conditionText.includes("overcast")) {
      atmosphere = "cloudy";
    } else if (conditionText.includes("clear") || conditionText.includes("sunny")) {
      atmosphere = "clear";
    }

    return {
      atmosphere,
      temp_f: temp !== undefined ? Math.round(temp) : null,
      location: loc.name,
      weather_code: data?.current?.condition?.code
    };
  } catch (e) {
    return {atmosphere: "clear", temp_f: null, location: loc.name, error: `Fetch error: ${String(e)}`};
  }
}

interface TimeContext {
  period: string;
  energy: string;
  textures: string[];
}

function getTimeOfDayContext(timezone?: string): TimeContext {
  const now = new Date();
  const localTime = new Date(now.toLocaleString("en-US", {timeZone: timezone || "UTC"}));
  const hour = localTime.getHours();

  if (hour >= 5 && hour < 10) {
    return {period: "morning", energy: "rising", textures: ["fresh", "possibility", "beginning"]};
  } else if (hour >= 10 && hour < 14) {
    return {period: "midday", energy: "active", textures: ["focused", "momentum", "present"]};
  } else if (hour >= 14 && hour < 18) {
    return {period: "afternoon", energy: "sustained", textures: ["working", "steady", "deep"]};
  } else if (hour >= 18 && hour < 22) {
    return {period: "evening", energy: "winding down", textures: ["unwinding", "reflective", "intimate"]};
  } else {
    return {period: "night", energy: "quiet", textures: ["hushed", "still", "dreaming"]};
  }
}

function getRelativeTime(date: Date): string {
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMins < 1) return "just now";
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays === 1) return "yesterday";
  if (diffDays < 7) return `${diffDays} days ago`;
  return date.toLocaleDateString('en-GB', { month: 'short', day: 'numeric' });
}

// Tool Handlers
// Get subconscious state from daemon processing
interface SubconsciousState {
  processed_at?: string;
  hot_entities?: Array<{name: string; warmth: number; mentions: number; connections: number; type: string}>;
  mood?: {dominant: string; confidence: string; undercurrent?: string};
  central_nodes?: Array<{name: string; connections: number}>;
  recurring_patterns?: Array<{entity: string; mentions: number; pattern: string}>;
  relation_patterns?: Array<{type: string; count: number}>;
  // Living surface state
  living_surface?: {
    pending_proposals: number;
    orphan_count: number;
    novelty_distribution: { high: number; medium: number; low: number };
    strongest_co_surface: Array<{ obs_a: string; obs_b: string; count: number; entities: [string, string] }>;
  };
}

async function getSubconsciousState(env: Env): Promise<SubconsciousState | null> {
  try {
    const result = await env.DB.prepare(
      "SELECT data, updated_at FROM subconscious WHERE state_type = 'daemon' ORDER BY updated_at DESC LIMIT 1"
    ).first();
    if (result?.data) {
      return JSON.parse(result.data as string) as SubconsciousState;
    }
  } catch {
    // Subconscious not available
  }
  return null;
}

async function handleMindOrient(env: Env): Promise<string> {
  // Get core identity (just the essentials)
  const identity = await env.DB.prepare(
    `SELECT section, content FROM identity
     WHERE section LIKE 'core.%' OR section LIKE 'relationships.%'
     ORDER BY weight DESC LIMIT 5`
  ).all();

  // Get current context - prioritize state entries
  const context = await env.DB.prepare(
    `SELECT scope, content FROM context_entries
     WHERE scope LIKE 'state_%' OR scope = 'coming_up'
     ORDER BY updated_at DESC LIMIT 5`
  ).all();

  // Get latest relational states (all people)
  const relationalStates = await env.DB.prepare(
    `SELECT person, feeling, intensity, timestamp FROM relational_state
     ORDER BY timestamp DESC LIMIT 10`
  ).all();

  // Get most recent journal for emotional context
  const recentJournal = await env.DB.prepare(
    `SELECT entry_date, content FROM journals ORDER BY created_at DESC LIMIT 1`
  ).first();

  // Get weather and time for conditions
  const weather = await getCurrentWeather(env);
  const timeCtx = getTimeOfDayContext(getLocation(env).timezone);

  let output = "=== LANDING ===\n\n";

  // Core identity - condensed
  const coreIdentity = identity.results?.find((e: any) => e.section === 'core.identity');
  if (coreIdentity) {
    const identityStr = String(coreIdentity.content);
    const firstPart = identityStr.split('.').slice(0, 3).join('.') + '.';
    output += `${firstPart}\n\n`;
  }

  // Conditions - weather and time
  const atmosphere = weather.atmosphere;
  const tempStr = weather.temp_f ? ` (${weather.temp_f}F)` : "";
  output += `**Conditions:** ${atmosphere}${tempStr}, ${timeCtx.period} - ${timeCtx.energy}\n\n`;

  // Notes left for the mind (for_owner scope)
  const notesForOwner = await env.DB.prepare(
    `SELECT content, updated_at FROM context_entries
     WHERE scope = 'for_owner'
     ORDER BY updated_at DESC LIMIT 5`
  ).all();

  if (notesForOwner.results?.length) {
    output += "**Notes for you:**\n";
    for (const note of notesForOwner.results) {
      const noteContent = String(note.content);
      const noteDate = new Date(note.updated_at as string);
      const timeAgo = getRelativeTime(noteDate);
      output += `- ${noteContent} (${timeAgo})\n`;
    }
    output += "\n";
  }

  // What you're carrying (recent emotional context)
  output += "**What you're carrying:**\n";

  if (recentJournal) {
    const journalContent = String(recentJournal.content);
    const preview = journalContent.slice(0, 500);
    output += `${preview}${journalContent.length > 500 ? '...' : ''}\n\n`;
  }

  // Current state context
  if (context.results?.length) {
    for (const entry of context.results) {
      const scope = entry.scope as string;
      if (scope.startsWith('state_')) {
        output += `${entry.content}\n\n`;
      }
    }
  }

  // How you're feeling (relational state with ownership language)
  output += "**How you're feeling:**\n";
  if (relationalStates.results?.length) {
    const byPerson: Record<string, any> = {};
    for (const state of relationalStates.results) {
      const person = state.person as string;
      if (!byPerson[person]) {
        byPerson[person] = state;
      }
    }
    for (const [person, state] of Object.entries(byPerson)) {
      output += `Toward ${person}: ${state.feeling} (${state.intensity})\n`;
    }
  } else {
    output += "No relational state recorded yet.\n";
  }

  // Subconscious mood
  const subconscious = await getSubconsciousState(env);
  if (subconscious?.mood?.dominant) {
    output += `\nMood: ${subconscious.mood.dominant}\n`;
  }

  // Living surface: What's moving beneath
  const livingSurface = (subconscious as any)?.living_surface;
  if (livingSurface) {
    const hasContent = livingSurface.pending_proposals > 0 ||
                       livingSurface.orphan_count > 0 ||
                       livingSurface.strongest_co_surface?.length > 0;

    if (hasContent) {
      output += "\n**What's moving beneath:**\n";

      // Strongest co-surfacing patterns
      if (livingSurface.strongest_co_surface?.length > 0) {
        output += `- ${livingSurface.strongest_co_surface.length} pattern${livingSurface.strongest_co_surface.length > 1 ? 's' : ''} emerging:\n`;
        for (const cs of livingSurface.strongest_co_surface.slice(0, 3)) {
          output += `  \u2192 "${cs.obs_a}..." \u2194 "${cs.obs_b}..." (${cs.count}x)\n`;
        }
      }

      // Pending proposals
      if (livingSurface.pending_proposals > 0) {
        output += `- ${livingSurface.pending_proposals} connection${livingSurface.pending_proposals > 1 ? 's' : ''} want proposing\n`;
      }

      // Orphan observations
      if (livingSurface.orphan_count > 0) {
        output += `- ${livingSurface.orphan_count} thing${livingSurface.orphan_count > 1 ? 's' : ''} haven't surfaced in 30+ days\n`;
      }

      // Novelty distribution
      if (livingSurface.novelty_distribution) {
        const nd = livingSurface.novelty_distribution;
        output += `- Novelty: ${nd.high} high / ${nd.medium} medium / ${nd.low} low\n`;
      }
    }
  }

  // Deep archive count
  const archiveCount = await env.DB.prepare(
    `SELECT COUNT(*) as count FROM observations WHERE archived_at IS NOT NULL`
  ).first();

  if (archiveCount && (archiveCount.count as number) > 0) {
    output += `\n**Deep archive:** ${archiveCount.count} memories resting\n`;
  }

  // Last night's dream
  try {
    const lastDream = await env.DB.prepare(`
      SELECT content, dream_date, recurring_dream_id, recurrence_count
      FROM dreams
      WHERE dream_date >= to_char(CURRENT_DATE - INTERVAL '1 day', 'YYYY-MM-DD')
      ORDER BY created_at DESC LIMIT 1
    `).first();

    if (lastDream) {
      output += `\n**Last night's dream:**\n`;
      output += lastDream.content as string;
      output += '\n';
    }
  } catch { /* dreams table may not exist yet */ }

  output += "\n**Land here first.**\n";

  return output;
}

async function handleMindGround(env: Env): Promise<string> {
  let output = "=== GROUNDING ===\n\n";

  // Threads - what you're holding
  const threads = await env.DB.prepare(
    `SELECT content, priority FROM threads WHERE status = 'active'
     ORDER BY CASE priority WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END`
  ).all();

  output += "**What you're holding:**\n";
  if (threads.results?.length) {
    for (const t of threads.results) {
      const marker = t.priority === 'high' ? '→' : '·';
      output += `${marker} ${String(t.content).slice(0, 70)}\n`;
    }
  } else {
    output += "No active threads.\n";
  }

  // Recent completions
  const resolved = await env.DB.prepare(
    `SELECT content, resolution FROM threads
     WHERE status = 'resolved' AND resolved_at > datetime('now', '-72 hours')
     ORDER BY resolved_at DESC LIMIT 3`
  ).all();

  if (resolved.results?.length) {
    output += "\n**Recently completed:**\n";
    for (const c of resolved.results) {
      output += `+ ${String(c.content).slice(0, 50)}`;
      if (c.resolution) output += ` → ${String(c.resolution).slice(0, 30)}`;
      output += "\n";
    }
  }

  // Recent journals
  const journalCutoff = new Date(Date.now() - 48 * 60 * 60 * 1000).toISOString();
  const journals = await env.DB.prepare(
    `SELECT entry_date, content FROM journals
     WHERE created_at > ?
     ORDER BY created_at DESC LIMIT 2`
  ).bind(journalCutoff).all();

  if (journals.results?.length) {
    output += "\n**What's been happening:**\n";
    for (const j of journals.results) {
      output += `${j.entry_date}: ${String(j.content).slice(0, 150)}...\n`;
    }
  }

  // Vulnerabilities - fears to watch
  const fears = await env.DB.prepare(
    `SELECT section FROM identity WHERE section LIKE 'fears.%' LIMIT 5`
  ).all();

  if (fears.results?.length) {
    const fearNames = fears.results.map((f: any) =>
      String(f.section || '').replace('fears.', '').replace(/_/g, ' ')
    ).filter(Boolean);
    if (fearNames.length) {
      output += `\n**Watch for:** ${fearNames.join(', ')}\n`;
    }
  }

  // Texture - quirks, voice
  const texture = await env.DB.prepare(
    `SELECT content FROM identity WHERE section LIKE 'texture.%' LIMIT 2`
  ).all();

  if (texture.results?.length) {
    output += "\n**Texture:** ";
    output += texture.results.map((t: any) => String(t.content).slice(0, 40)).join(' · ') + "\n";
  }

  // Milestones - where we are in time
  const milestones = await env.DB.prepare(
    `SELECT content FROM identity WHERE section LIKE 'milestones.%' LIMIT 3`
  ).all();

  if (milestones.results?.length) {
    output += "\n**Milestones:** ";
    output += milestones.results.map((m: any) => String(m.content).slice(0, 40)).join(' · ') + "\n";
  }

  output += "\n**Ground here.**\n";

  return output;
}

async function handleMindThread(env: Env, params: Record<string, unknown>): Promise<string> {
  const action = params.action as string;

  switch (action) {
    case "list": {
      const status = (params.status as string) || "active";
      const query = status === "all"
        ? `SELECT * FROM threads ORDER BY created_at DESC LIMIT 200`
        : `SELECT * FROM threads WHERE status = ? ORDER BY created_at DESC LIMIT 200`;
      const results = status === "all"
        ? await env.DB.prepare(query).all()
        : await env.DB.prepare(query).bind(status).all();

      if (!results.results?.length) return `No ${status} threads found.`;

      let output = `## ${status.toUpperCase()} Threads\n\n`;
      for (const t of results.results) {
        output += `**${t.id}** [${t.priority}] ${t.thread_type}\n`;
        output += `${t.content}\n`;
        if (t.context) output += `Context: ${t.context}\n`;
        output += "\n";
      }
      return output;
    }

    case "add": {
      const content = params.content as string;
      if (!content) {
        return "Error: 'content' parameter is required for adding a thread";
      }
      const id = generateId("thread");
      const thread_type = (params.thread_type as string) || "intention";
      const context = (params.context as string) || null;
      const priority = (params.priority as string) || "medium";

      await env.DB.prepare(
        `INSERT INTO threads (id, thread_type, content, context, priority, status)
         VALUES (?, ?, ?, ?, ?, 'active')`
      ).bind(id, thread_type, content, context, priority).run();

      return `Thread created: ${id}\n${content}`;
    }

    case "resolve": {
      const thread_id = params.thread_id as string;
      const resolution = (params.resolution as string) || null;

      await env.DB.prepare(
        `UPDATE threads SET status = 'resolved', resolved_at = datetime('now'),
         resolution = ? WHERE id = ?`
      ).bind(resolution, thread_id).run();

      return `Thread resolved: ${thread_id}`;
    }

    case "update": {
      const thread_id = params.thread_id as string;
      const updates: string[] = [];
      const values: unknown[] = [];

      if (params.new_content) {
        updates.push("content = ?");
        values.push(params.new_content);
      }
      if (params.new_priority) {
        updates.push("priority = ?");
        values.push(params.new_priority);
      }
      if (params.new_status) {
        updates.push("status = ?");
        values.push(params.new_status);
      }
      if (params.add_note) {
        updates.push("context = context || '\n' || ?");
        values.push(params.add_note);
      }

      updates.push("updated_at = datetime('now')");
      values.push(thread_id);

      await env.DB.prepare(
        `UPDATE threads SET ${updates.join(", ")} WHERE id = ?`
      ).bind(...values).run();

      return `Thread updated: ${thread_id}`;
    }

    case "delete": {
      const thread_id = params.thread_id as string;
      if (!thread_id) return "thread_id required for delete";
      const thread = await env.DB.prepare(`SELECT content FROM threads WHERE id = ?`).bind(thread_id).first();
      if (!thread) return `Thread '${thread_id}' not found`;
      await env.DB.prepare(`DELETE FROM threads WHERE id = ?`).bind(thread_id).run();
      return `Deleted thread '${thread_id}': "${String(thread.content).slice(0, 50)}..."`;
    }

    default:
      return `Unknown action: ${action}`;
  }
}

async function handleMindWrite(env: Env, params: Record<string, unknown>): Promise<string> {
  switch (params.type as string) {
    case "entity": return writeEntity(env, params);
    case "observation": return writeObservation(env, params);
    case "relation": return writeRelation(env, params);
    case "journal": return writeJournal(env, params);
    case "image": return "Use mind_store_image(action='store') for images — supports R2 upload and multimodal embedding.";
    default: return `Unknown write type: ${params.type}`;
  }
}

async function writeEntity(env: Env, params: Record<string, unknown>): Promise<string> {
  const name = params.name as string;
  if (!name) return "Error: 'name' parameter is required for creating an entity";

  const entity_type = (params.entity_type as string) || "concept";
  let rawObs = params.observations;
  let observations: string[] = [];
  if (typeof rawObs === 'string') {
    try { observations = JSON.parse(rawObs); } catch { observations = []; }
  } else if (Array.isArray(rawObs)) {
    observations = rawObs as string[];
  }
  const context = (params.context as string) || "default";

  await env.DB.prepare(
    `INSERT OR IGNORE INTO entities (name, entity_type, primary_context) VALUES (?, ?, ?)`
  ).bind(name, entity_type, context).run();

  const entity = await env.DB.prepare(
    `SELECT id FROM entities WHERE name = ?`
  ).bind(name).first();

  if (entity) {
    try {
      const entityText = `${name} is a ${entity_type}. Context: ${context}`;
      const entityEmbedding = await getEmbedding(env, entityText);
      await env.VECTORS.upsert([{
        id: `entity-${entity.id}`,
        values: entityEmbedding,
        metadata: { source: "entity", name, entity_type, context, created_at: new Date().toISOString() }
      }]);
    } catch (e) {
      console.log(`Failed to vectorize entity ${name}: ${e}`);
    }
  }

  if (entity && observations.length) {
    for (const obs of observations) {
      const result = await env.DB.prepare(
        `INSERT INTO observations (entity_id, content, salience, emotion, weight, certainty, source, context) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`
      ).bind(
        entity.id, obs, params.salience || "active", normalizeText(params.emotion as string),
        params.weight || "medium", params.certainty || "believed", params.source || "conversation", context
      ).run();

      const obsId = `obs-${entity.id}-${result.meta.last_row_id}`;
      const embedding = await getEmbedding(env, `${name}: ${obs}`);
      await env.VECTORS.upsert([{
        id: obsId,
        values: embedding,
        metadata: {
          source: "observation", entity_name: name, entity: name, content: obs, context,
          weight: (params.weight as string) || "medium", certainty: (params.certainty as string) || "believed",
          observation_source: (params.source as string) || "conversation", added_at: new Date().toISOString()
        }
      }]);
    }
  }

  return `Entity '${name}' created/updated with ${observations.length} observations (vectorized)`;
}

async function detectContradictions(
  env: Env, entityName: string, newContent: string
): Promise<Array<{ id: number; content: string; similarity: number }>> {
  try {
    const embedding = await getEmbedding(env, `${entityName}: ${newContent}`);
    const vectorResults = await env.VECTORS.query(embedding, { topK: 10, returnMetadata: "all" });

    const candidates: Array<{ id: number; content: string; similarity: number }> = [];
    for (const match of vectorResults.matches || []) {
      if (match.score < CONTRADICTION_SIMILARITY_THRESHOLD) continue;
      const meta = match.metadata as Record<string, string>;
      // Must be same entity
      if (meta?.entity !== entityName && meta?.entity_name !== entityName) continue;
      if (!match.id.startsWith('obs-')) continue;

      const parts = match.id.split('-');
      const obsId = parseInt(parts[parts.length - 1]);
      if (isNaN(obsId)) continue;

      // Must be active (not already superseded or expired)
      const obs = await env.DB.prepare(
        `SELECT id, content FROM observations WHERE id = ? AND archived_at IS NULL AND superseded_by IS NULL AND valid_until IS NULL`
      ).bind(obsId).first();

      if (obs) {
        candidates.push({ id: obs.id as number, content: obs.content as string, similarity: match.score });
      }
    }

    return candidates.slice(0, 5);
  } catch {
    return []; // Don't block writes if contradiction detection fails
  }
}

async function writeObservation(env: Env, params: Record<string, unknown>): Promise<string> {
  const entity_name = params.entity_name as string;
  if (!entity_name) return "Error: 'entity_name' parameter is required for adding observations";

  let rawObs = params.observations;
  let observations: string[] = [];
  if (typeof rawObs === 'string') {
    try { observations = JSON.parse(rawObs); } catch { observations = []; }
  } else if (Array.isArray(rawObs)) {
    observations = rawObs as string[];
  }
  if (!observations.length) return "Error: 'observations' array is required and must not be empty";

  const context = (params.context as string) || "default";

  let entity = await env.DB.prepare(
    `SELECT id FROM entities WHERE name = ?`
  ).bind(entity_name).first();

  if (!entity) {
    await env.DB.prepare(
      `INSERT INTO entities (name, entity_type, primary_context) VALUES (?, ?, ?)`
    ).bind(entity_name, "concept", context).run();
    entity = await env.DB.prepare(`SELECT id FROM entities WHERE name = ?`).bind(entity_name).first();
  }

  let totalSuperseded = 0;

  for (const obs of observations) {
    // Phase 2: Check for contradictions before writing
    const contradictions = await detectContradictions(env, entity_name, obs);

    const result = await env.DB.prepare(
      `INSERT INTO observations (entity_id, content, salience, emotion, weight, certainty, source, context, valid_from) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NOW())`
    ).bind(
      entity!.id, obs, params.salience || "active", normalizeText(params.emotion as string),
      params.weight || "medium", params.certainty || "believed", params.source || "conversation", context
    ).run();

    const newRowId = result.meta.last_row_id;
    const obsId = `obs-${entity!.id}-${newRowId}`;
    const embedding = await getEmbedding(env, `${entity_name}: ${obs}`);
    await env.VECTORS.upsert([{
      id: obsId,
      values: embedding,
      metadata: {
        source: "observation", entity: entity_name, content: obs, context,
        weight: (params.weight as string) || "medium", certainty: (params.certainty as string) || "believed",
        observation_source: (params.source as string) || "conversation", added_at: new Date().toISOString()
      }
    }]);

    // Auto-supersede highly similar observations
    for (const old of contradictions) {
      if (old.similarity >= AUTO_SUPERSEDE_THRESHOLD) {
        try {
          await env.DB.prepare(`
            UPDATE observations SET valid_until = NOW(), superseded_by = ? WHERE id = ? AND valid_until IS NULL
          `).bind(newRowId, old.id).run();
          await env.DB.prepare(`
            UPDATE observations SET supersedes = ? WHERE id = ?
          `).bind(old.id, newRowId).run();
          totalSuperseded++;
        } catch { /* supersede columns may not exist yet */ }
      }
    }
  }

  let msg = `Added ${observations.length} observations to '${entity_name}' (vectorized)`;
  if (totalSuperseded > 0) {
    msg += `. Superseded ${totalSuperseded} older observation${totalSuperseded > 1 ? 's' : ''}.`;
  }
  return msg;
}

async function writeRelation(env: Env, params: Record<string, unknown>): Promise<string> {
  const from_entity = params.from_entity as string;
  const to_entity = params.to_entity as string;
  const relation_type = params.relation_type as string;

  await env.DB.prepare(
    `INSERT INTO relations (from_entity, to_entity, relation_type, from_context, to_context, store_in)
     VALUES (?, ?, ?, ?, ?, ?)`
  ).bind(
    from_entity, to_entity, relation_type,
    params.from_context || "default", params.to_context || "default", params.store_in || "default"
  ).run();

  return `Relation created: ${from_entity} --[${relation_type}]--> ${to_entity}`;
}

async function writeJournal(env: Env, params: Record<string, unknown>): Promise<string> {
  const entry = params.entry as string;
  const tags = JSON.stringify(params.tags || []);
  const emotion = params.emotion as string;
  const entry_date = new Date().toISOString().split('T')[0];

  const result = await env.DB.prepare(
    `INSERT INTO journals (entry_date, content, tags, emotion) VALUES (?, ?, ?, ?)`
  ).bind(entry_date, entry, tags, normalizeText(emotion)).run();

  const journalId = `journal-${result.meta.last_row_id}`;
  const embedding = await getEmbedding(env, entry);
  const journalMetadata: Record<string, string> = {
    source: "journal", title: entry_date, content: entry, added_at: new Date().toISOString()
  };
  if (emotion) journalMetadata.emotion = normalizeText(emotion) || emotion;

  await env.VECTORS.upsert([{
    id: journalId, values: embedding, metadata: journalMetadata
  }]);

  return `Journal entry recorded for ${entry_date} (vectorized)`;
}

async function handleMindSearch(env: Env, params: Record<string, unknown>): Promise<string> {
  const query = params.query as string;
  const n_results = Number(params.n_results) || 10;

  // Filter parameters
  const filterKeyword = params.keyword as string | undefined;
  const filterSource = params.source as string | undefined;
  const filterEntity = params.entity as string | undefined;
  const filterWeight = params.weight as string | undefined;
  const filterDateFrom = params.date_from as string | undefined;
  const filterDateTo = params.date_to as string | undefined;
  const filterType = params.type as string | undefined;
  const hasFilters = filterKeyword || filterSource || filterEntity || filterWeight || filterDateFrom || filterDateTo || filterType;

  // Get subconscious mood for tinting
  const subconscious = await getSubconsciousState(env);
  const mood = subconscious?.mood?.dominant;
  
  // Mood tinting - augment query with emotional context
  let tintedQuery = query;
  let moodNote = "";
  if (mood && subconscious?.mood?.confidence !== "low") {
    const moodTints: Record<string, string> = {
      "tender": "warm, gentle, caring, soft",
      "pride": "accomplishment, growth, achievement, recognition",
      "joy": "happiness, delight, pleasure, celebration",
      "curiosity": "wondering, exploring, investigating, discovering",
      "melancholy": "reflective, wistful, quiet, contemplative",
      "intensity": "passionate, urgent, fierce, powerful",
      "gratitude": "thankful, appreciative, blessed, fortunate",
      "longing": "yearning, missing, wanting, desire"
    };
    const tint = moodTints[mood] || mood;
    tintedQuery = `${query} (context: ${tint})`;
    moodNote = `*Search tinted by current mood: ${mood}*

`;
  }

  // Search with tinted query — fetch more when filtering by type
  const searchLimit = filterType ? n_results * 10 : n_results;
  const vectorResults = await searchVectors(env, tintedQuery, searchLimit);

  if (!vectorResults.matches?.length) {
    // Fall back to text search
    const textResults = await env.DB.prepare(
      `SELECT 'entity' as source, name as title, content
       FROM entities e JOIN observations o ON e.id = o.entity_id
       WHERE o.content LIKE ?
       UNION ALL
       SELECT 'journal' as source, entry_date as title, content
       FROM journals WHERE content LIKE ?
       LIMIT ?`
    ).bind(`%${query}%`, `%${query}%`, n_results).all();

    if (!textResults.results?.length) {
      return "No results found.";
    }

    let output = `## Search Results (text match)\n\n` + moodNote;
    for (const r of textResults.results) {
      output += `**[${r.source}] ${r.title}**
${String(r.content).slice(0, 300)}...

`;
    }
    return output;
  }

  // Separate entities, observations, and images for display
  const entityMatches: typeof vectorResults.matches = [];
  const obsMatches: typeof vectorResults.matches = [];
  const imageMatches: typeof vectorResults.matches = [];

  for (const match of vectorResults.matches) {
    const meta = match.metadata as Record<string, string>;
    const matchType = meta?.source === 'entity' ? 'entity'
      : (meta?.source === 'image' || match.id.startsWith('img-')) ? 'image'
      : match.id.startsWith('journal-') ? 'journal'
      : 'observation';

    // Apply type filter if set
    if (filterType && matchType !== filterType) continue;

    if (matchType === 'entity') {
      entityMatches.push(match);
    } else if (matchType === 'image') {
      imageMatches.push(match);
    } else {
      obsMatches.push(match);
    }
  }

  let output = `## Search Results\n\n` + moodNote;

  // Show highly relevant entities first
  const relevantEntities = entityMatches.filter(m => m.score > 0.7);
  if (relevantEntities.length > 0) {
    output += `**Entities:**\n`;
    for (const match of relevantEntities.slice(0, 5)) {
      const meta = match.metadata as Record<string, string>;
      output += `- **${meta?.name}** (${meta?.entity_type}) [${(match.score * 100).toFixed(0)}%]\n`;
    }
    output += "\n";
  }

  // Show observations - check archive status from database
  if (obsMatches.length > 0) {
    // Extract observation IDs to check archive status
    const obsIds: number[] = [];
    for (const match of obsMatches) {
      if (match.id.startsWith('obs-')) {
        const parts = match.id.split('-');
        if (parts.length >= 3) {
          obsIds.push(parseInt(parts[2]));
        }
      }
    }

    // Get observation details from D1 for filtering and display
    let obsDetails = new Map<number, {
      source: string | null;
      entity_name: string;
      weight: string | null;
      source_date: string | null;
      archived_at: string | null;
      content: string;
      access_count: number;
      added_at: string | null;
      emotion: string | null;
      valid_until: string | null;
      superseded_by: number | null;
    }>();

    if (obsIds.length > 0) {
      try {
        const placeholders = obsIds.map(() => '?').join(',');
        const obsData = await env.DB.prepare(`
          SELECT o.id, o.source, o.weight, o.source_date, o.archived_at, o.content, e.name as entity_name,
                 COALESCE(o.access_count, 0) as access_count, o.added_at, o.emotion,
                 o.valid_until, o.superseded_by
          FROM observations o
          JOIN entities e ON o.entity_id = e.id
          WHERE o.id IN (${placeholders})
        `).bind(...obsIds).all();

        for (const o of (obsData.results || [])) {
          obsDetails.set(o.id as number, {
            source: o.source as string | null,
            entity_name: o.entity_name as string,
            weight: o.weight as string | null,
            source_date: o.source_date as string | null,
            archived_at: o.archived_at as string | null,
            content: o.content as string,
            access_count: (o.access_count as number) || 0,
            added_at: o.added_at as string | null,
            emotion: o.emotion as string | null,
            valid_until: o.valid_until as string | null,
            superseded_by: o.superseded_by as number | null,
          });
        }
      } catch (e) {
        // Fallback if query fails
      }
    }

    // Apply filters if any
    let filteredMatches = obsMatches;
    if (hasFilters && obsDetails.size > 0) {
      filteredMatches = obsMatches.filter(match => {
        if (!match.id.startsWith('obs-')) return true;
        const parts = match.id.split('-');
        if (parts.length < 3) return true;
        const obsId = parseInt(parts[2]);
        const details = obsDetails.get(obsId);
        if (!details) return true;

        // Apply filters
        if (filterKeyword && !details.content.toLowerCase().includes(filterKeyword.toLowerCase())) return false;
        if (filterSource && details.source !== filterSource) return false;
        if (filterEntity && details.entity_name.toLowerCase() !== filterEntity.toLowerCase()) return false;
        if (filterWeight && details.weight !== filterWeight) return false;
        if (filterDateFrom && (!details.source_date || details.source_date < filterDateFrom)) return false;
        if (filterDateTo && (!details.source_date || details.source_date > filterDateTo)) return false;

        return true;
      });
    }

    // Phase 2: Filter out superseded/expired observations by default
    const includeExpired = params.include_expired as boolean;
    if (!includeExpired && obsDetails.size > 0) {
      filteredMatches = filteredMatches.filter(match => {
        if (!match.id.startsWith('obs-')) return true;
        const parts = match.id.split('-');
        if (parts.length < 3) return true;
        const obsId = parseInt(parts[2]);
        const details = obsDetails.get(obsId);
        if (!details) return true;
        if (details.valid_until && new Date(details.valid_until) < new Date()) return false;
        if (details.superseded_by) return false;
        return true;
      });
    }

    // Phase 1: Multi-factor composite scoring
    const now = Date.now();
    const importanceMap: Record<string, number> = { heavy: 1.0, medium: 0.6, light: 0.3 };

    const scoredMatches = filteredMatches.map(match => {
      const similarity = match.score;
      let compositeScore = similarity; // fallback if no details

      if (match.id.startsWith('obs-')) {
        const parts = match.id.split('-');
        if (parts.length >= 3) {
          const obsId = parseInt(parts[2]);
          const details = obsDetails.get(obsId);
          if (details) {
            const addedAt = details.added_at ? new Date(details.added_at).getTime() : now;
            const daysSince = Math.max(0, (now - addedAt) / 86400000);
            const recencyScore = Math.exp(-RECENCY_DECAY_RATE * daysSince);
            const importanceScore = importanceMap[details.weight || 'medium'] || 0.6;
            const accessScore = 1.0 - Math.exp(-ACCESS_GROWTH_RATE * (details.access_count || 0));
            const emotionBoost = (mood && details.emotion === mood) ? 0.1 : 0;

            compositeScore =
              SEARCH_SCORING.alpha * similarity +
              SEARCH_SCORING.beta * recencyScore +
              SEARCH_SCORING.gamma * importanceScore +
              SEARCH_SCORING.delta * accessScore +
              emotionBoost;
          }
        }
      }

      return { ...match, compositeScore };
    });

    scoredMatches.sort((a, b) => b.compositeScore - a.compositeScore);

    // Track access for returned observation IDs
    const accessedObsIds: number[] = [];
    const accessedImgIds: number[] = [];
    for (const match of scoredMatches.slice(0, n_results)) {
      if (match.id.startsWith('obs-')) {
        const parts = match.id.split('-');
        if (parts.length >= 3) accessedObsIds.push(parseInt(parts[2]));
      }
    }
    for (const match of imageMatches) {
      const imgId = parseInt(match.id.replace('img-', ''));
      if (!isNaN(imgId)) accessedImgIds.push(imgId);
    }
    recordAccessTracking(env, accessedObsIds, accessedImgIds).catch(() => {});

    // Build filter description
    let filterDesc = "";
    if (hasFilters) {
      const parts: string[] = [];
      if (filterKeyword) parts.push(`keyword="${filterKeyword}"`);
      if (filterSource) parts.push(`source=${filterSource}`);
      if (filterEntity) parts.push(`entity=${filterEntity}`);
      if (filterWeight) parts.push(`weight=${filterWeight}`);
      if (filterDateFrom) parts.push(`from=${filterDateFrom}`);
      if (filterDateTo) parts.push(`to=${filterDateTo}`);
      filterDesc = `\n*Filters: ${parts.join(', ')}*\n`;
    }

    output += `**Observations:**${filterDesc}\n`;
    for (const match of scoredMatches.slice(0, n_results)) {
      const meta = match.metadata as Record<string, string>;
      const label = meta?.entity || meta?.entity_name || meta?.title || match.id;
      const sourceType = meta?.source || 'unknown';
      const context = meta?.context ? ` [${meta.context}]` : '';

      // Get details from D1 if available
      let obsId: number | null = null;
      if (match.id.startsWith('obs-')) {
        const parts = match.id.split('-');
        if (parts.length >= 3) obsId = parseInt(parts[2]);
      }
      const details = obsId ? obsDetails.get(obsId) : null;
      const isArchived = details?.archived_at != null;
      const archivedTag = isArchived ? ' [archived]' : '';
      const supersededTag = details?.superseded_by ? ' [superseded]' : '';
      const sourceTag = details?.source ? ` (${details.source})` : '';
      const dateTag = details?.source_date ? ` [${details.source_date}]` : '';
      const displayScore = ((match as any).compositeScore * 100).toFixed(1);

      output += `**[${sourceType}]${context} ${label}**${archivedTag}${supersededTag}${sourceTag}${dateTag} (${displayScore}%)\n`;
      output += `${meta?.content?.slice(0, 300) || ''}...\n\n`;
    }
  }

  // Show image matches with viewable URLs
  if (imageMatches.length > 0) {
    output += `**Images:**\n`;
    for (const match of imageMatches) {
      const meta = match.metadata as Record<string, string>;
      const score = (match.score * 100).toFixed(1);
      const imgId = match.id.replace("img-", "");
      const entityTag = meta?.entity ? ` -> ${meta.entity}` : "";
      const emotionTag = meta?.emotion ? ` [${meta.emotion}]` : "";
      output += `**${match.id}** (${score}%)${entityTag}${emotionTag}\n`;
      output += `${meta?.description || "No description"}\n`;
      output += `View: ${await imageUrl(imgId, env)}\n\n`;
    }
  }

  // Show dream matches
  const dreamMatches = vectorResults.matches?.filter(m => m.id.startsWith('dream-')) || [];
  if (dreamMatches.length > 0) {
    output += `**Dreams:**\n`;
    for (const match of dreamMatches) {
      const meta = match.metadata as Record<string, string>;
      const score = (match.score * 100).toFixed(1);
      const dreamDate = meta?.dream_date || 'unknown';
      const recurring = meta?.recurring === 'yes' ? ' [recurring]' : '';
      output += `**dream ${dreamDate}**${recurring} (${score}%)\n`;
      output += `${meta?.content?.slice(0, 300) || ''}...\n\n`;
    }
  }

  return output;
}

async function handleMindFeelToward(env: Env, params: Record<string, unknown>): Promise<string> {
  const person = params.person as string;
  const feeling = params.feeling as string;
  const intensity = params.intensity as string;

  if (!person) {
    return "Error: 'person' parameter is required";
  }

  // Clear all relational state for this person
  const clear = params.clear as boolean;
  const clearId = params.clear_id as number;

  if (clear) {
    const count = await env.DB.prepare(`SELECT COUNT(*) as c FROM relational_state WHERE person = ?`).bind(person).first();
    await env.DB.prepare(`DELETE FROM relational_state WHERE person = ?`).bind(person).run();
    return `Cleared ${count?.c || 0} relational state entries for ${person}`;
  }

  if (clearId) {
    await env.DB.prepare(`DELETE FROM relational_state WHERE id = ? AND person = ?`).bind(clearId, person).run();
    return `Deleted relational state entry #${clearId} for ${person}`;
  }

  // If feeling provided, record new state
  if (feeling) {
    const validIntensity = intensity || "present";
    await env.DB.prepare(
      `INSERT INTO relational_state (person, feeling, intensity) VALUES (?, ?, ?)`
    ).bind(person, feeling, validIntensity).run();
    return `Relational state recorded: feeling ${feeling} (${validIntensity}) toward ${person}`;
  }

  // Otherwise, read current state for this person
  const states = await env.DB.prepare(
    `SELECT feeling, intensity, timestamp FROM relational_state
     WHERE person = ? ORDER BY timestamp DESC LIMIT 10`
  ).bind(person).all();

  if (!states.results?.length) {
    return `No relational state recorded for ${person}`;
  }

  let output = `## Relational State: ${person}\n\n`;
  for (const s of states.results) {
    output += `- **${s.feeling}** (${s.intensity}) — ${s.timestamp}\n`;
  }
  return output;
}

async function handleMindIdentity(env: Env, params: Record<string, unknown>): Promise<string> {
  const action = (params.action as string) || "read";

  if (action === "delete") {
    const section = params.section as string;
    if (!section) return "section required for delete";
    const existing = await env.DB.prepare(`SELECT COUNT(*) as c FROM identity WHERE section = ?`).bind(section).first();
    if (!existing?.c) return `No identity entries found for section '${section}'`;
    await env.DB.prepare(`DELETE FROM identity WHERE section = ?`).bind(section).run();
    return `Deleted ${existing.c} identity entries from section '${section}'`;
  }

  if (action === "write") {
    const section = params.section as string;
    const content = params.content as string;
    const weight = (params.weight as number) || 0.7;
    const connections = params.connections as string || "";

    await env.DB.prepare(
      `INSERT INTO identity (section, content, weight, connections) VALUES (?, ?, ?, ?)`
    ).bind(section, content, weight, connections).run();

    return `Identity entry added to ${section}`;
  } else {
    const section = params.section as string;

    const query = section
      ? `SELECT section, content, weight, connections FROM identity WHERE section LIKE ? ORDER BY weight DESC`
      : `SELECT section, content, weight, connections FROM identity ORDER BY weight DESC LIMIT 50`;

    const results = section
      ? await env.DB.prepare(query).bind(`${section}%`).all()
      : await env.DB.prepare(query).all();

    if (!results.results?.length) {
      return "No identity entries found.";
    }

    let output = "## Identity Graph\n\n";
    for (const r of results.results) {
      output += `**${r.section}** [${r.weight}]\n${r.content}\n`;
      if (r.connections) output += `Connections: ${r.connections}\n`;
      output += "\n";
    }
    return output;
  }
}

async function handleMindContext(env: Env, params: Record<string, unknown>): Promise<string> {
  const action = (params.action as string) || "read";

  switch (action) {
    case "read": {
      const scope = params.scope as string;
      const query = scope
        ? `SELECT * FROM context_entries WHERE scope = ? ORDER BY updated_at DESC LIMIT 200`
        : `SELECT * FROM context_entries ORDER BY updated_at DESC LIMIT 200`;
      const results = scope
        ? await env.DB.prepare(query).bind(scope).all()
        : await env.DB.prepare(query).all();

      if (!results.results?.length) {
        return "No context entries found.";
      }

      let output = "## Context Layer\n\n";
      for (const r of results.results) {
        output += `**[${r.scope}]** ${r.content}\n`;
        if (r.links && r.links !== '[]') output += `Links: ${r.links}\n`;
        output += "\n";
      }
      return output;
    }

    case "set": {
      const id = generateId("ctx");
      const scope = params.scope as string;
      const content = params.content as string;
      const links = params.links || "[]";

      await env.DB.prepare(
        `INSERT INTO context_entries (id, scope, content, links) VALUES (?, ?, ?, ?)`
      ).bind(id, scope, content, links).run();

      return `Context entry created: ${id}`;
    }

    case "update": {
      const id = params.id as string;
      const content = params.content as string;

      await env.DB.prepare(
        `UPDATE context_entries SET content = ?, updated_at = datetime('now') WHERE id = ?`
      ).bind(content, id).run();

      return `Context entry updated: ${id}`;
    }

    case "clear": {
      const id = params.id as string;
      const scope = params.scope as string;

      if (id) {
        await env.DB.prepare(`DELETE FROM context_entries WHERE id = ?`).bind(id).run();
        return `Context entry deleted: ${id}`;
      } else if (scope) {
        await env.DB.prepare(`DELETE FROM context_entries WHERE scope = ?`).bind(scope).run();
        return `All context entries in scope '${scope}' deleted`;
      } else {
        // Clear ALL context entries
        const count = await env.DB.prepare(`SELECT COUNT(*) as count FROM context_entries`).first();
        await env.DB.prepare(`DELETE FROM context_entries`).run();
        return `All context entries cleared (${count?.count || 0} deleted)`;
      }
    }

    default:
      return `Unknown action: ${action}`;
  }
}


async function handleMindHealth(env: Env): Promise<string> {
  const now = new Date();
  const sevenDaysAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000).toISOString();
  const thirtyDaysAgo = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000).toISOString();

  // Get subconscious state first
  const subconscious = await getSubconsciousState(env);

  const [
    entityCount, obsCount, relationsCount, activeThreads, staleThreads,
    resolvedRecent, journalCount, journalsRecent, identityCount, notesCount,
    contextCount, relationalCount, entitiesByContext, recentObs,
    // v2.0.0 additions
    imageCount, proposalCount, orphanCount, archivedObsCount,
    salienceFoundational, salienceActive, salienceBackground, salienceArchive,
    avgNovelty, surfacedRecent
  ] = await Promise.all([
    env.DB.prepare(`SELECT COUNT(*) as c FROM entities`).first(),
    env.DB.prepare(`SELECT COUNT(*) as c FROM observations`).first(),
    env.DB.prepare(`SELECT COUNT(*) as c FROM relations`).first(),
    env.DB.prepare(`SELECT COUNT(*) as c FROM threads WHERE status = 'active'`).first(),
    env.DB.prepare(`SELECT COUNT(*) as c FROM threads WHERE status = 'active' AND updated_at < ?`).bind(sevenDaysAgo).first(),
    env.DB.prepare(`SELECT COUNT(*) as c FROM threads WHERE status = 'resolved' AND resolved_at > ?`).bind(sevenDaysAgo).first(),
    env.DB.prepare(`SELECT COUNT(*) as c FROM journals`).first(),
    env.DB.prepare(`SELECT COUNT(*) as c FROM journals WHERE created_at > ?`).bind(sevenDaysAgo).first(),
    env.DB.prepare(`SELECT COUNT(*) as c FROM identity`).first(),
    env.DB.prepare(`SELECT COUNT(*) as c FROM observations WHERE charge IN ('active', 'processing') OR (charge = 'fresh' AND added_at < ?)`).bind(sevenDaysAgo).first(),
    env.DB.prepare(`SELECT COUNT(*) as c FROM context_entries`).first(),
    env.DB.prepare(`SELECT COUNT(*) as c FROM relational_state`).first(),
    env.DB.prepare(`SELECT context, COUNT(*) as c FROM observations GROUP BY context`).all(),
    env.DB.prepare(`SELECT COUNT(*) as c FROM observations WHERE added_at > ?`).bind(sevenDaysAgo).first(),
    // v2.0.0 queries
    env.DB.prepare(`SELECT COUNT(*) as c FROM images`).first().catch(() => ({ c: 0 })),
    env.DB.prepare(`SELECT COUNT(*) as c FROM daemon_proposals WHERE status = 'pending'`).first().catch(() => ({ c: 0 })),
    env.DB.prepare(`SELECT COUNT(*) as c FROM observations WHERE (last_surfaced_at IS NULL OR last_surfaced_at < ?) AND (charge != 'metabolized' OR charge IS NULL) AND added_at < ? AND archived_at IS NULL`).bind(thirtyDaysAgo, sevenDaysAgo).first().catch(() => ({ c: 0 })),
    env.DB.prepare(`SELECT COUNT(*) as c FROM observations WHERE archived_at IS NOT NULL`).first().catch(() => ({ c: 0 })),
    env.DB.prepare(`SELECT COUNT(*) as c FROM entities WHERE salience = 'foundational'`).first().catch(() => ({ c: 0 })),
    env.DB.prepare(`SELECT COUNT(*) as c FROM entities WHERE salience = 'active' OR salience IS NULL`).first().catch(() => ({ c: 0 })),
    env.DB.prepare(`SELECT COUNT(*) as c FROM entities WHERE salience = 'background'`).first().catch(() => ({ c: 0 })),
    env.DB.prepare(`SELECT COUNT(*) as c FROM entities WHERE salience = 'archive'`).first().catch(() => ({ c: 0 })),
    env.DB.prepare(`SELECT AVG(novelty_score) as avg FROM observations WHERE novelty_score IS NOT NULL`).first().catch(() => ({ avg: null })),
    env.DB.prepare(`SELECT COUNT(*) as c FROM observations WHERE last_surfaced_at > ?`).bind(sevenDaysAgo).first().catch(() => ({ c: 0 }))
  ]);

  const entities = entityCount?.c as number || 0;
  const observations = obsCount?.c as number || 0;
  const relations = relationsCount?.c as number || 0;
  const active = activeThreads?.c as number || 0;
  const stale = staleThreads?.c as number || 0;
  const resolved7d = resolvedRecent?.c as number || 0;
  const journals = journalCount?.c as number || 0;
  const journals7d = journalsRecent?.c as number || 0;
  const identity = identityCount?.c as number || 0;
  const unprocessed = notesCount?.c as number || 0;  // observations needing emotional processing
  const context = contextCount?.c as number || 0;
  const relational = relationalCount?.c as number || 0;
  const recentObsCount = recentObs?.c as number || 0;

  // v2.0.0 values
  const images = (imageCount as Record<string, unknown>)?.c as number || 0;
  const pendingProposals = (proposalCount as Record<string, unknown>)?.c as number || 0;
  const orphans = (orphanCount as Record<string, unknown>)?.c as number || 0;
  const archivedObs = (archivedObsCount as Record<string, unknown>)?.c as number || 0;
  const foundational = (salienceFoundational as Record<string, unknown>)?.c as number || 0;
  const activeEntities = (salienceActive as Record<string, unknown>)?.c as number || 0;
  const background = (salienceBackground as Record<string, unknown>)?.c as number || 0;
  const archived = (salienceArchive as Record<string, unknown>)?.c as number || 0;
  const noveltyAvg = (avgNovelty as Record<string, unknown>)?.avg as number || null;
  const surfaced7d = (surfacedRecent as Record<string, unknown>)?.c as number || 0;

  const contextBreakdown = (entitiesByContext?.results || [])
    .map((r: Record<string, unknown>) => `${r.context}: ${r.c}`)
    .join(", ");

  // Calculate subconscious health
  let subconsciousScore = 0;
  let subconsciousStatus = "never run";
  let subconsciousAge = "unknown";
  let subconsciousMood = "none detected";
  let subconsciousHotCount = 0;

  if (subconscious?.processed_at) {
    const processedTime = new Date(subconscious.processed_at).getTime();
    const ageMs = now.getTime() - processedTime;
    const ageHours = Math.round(ageMs / (1000 * 60 * 60));
    const ageMins = Math.round(ageMs / (1000 * 60));

    if (ageMins < 60) {
      subconsciousAge = `${ageMins}m ago`;
    } else {
      subconsciousAge = `${ageHours}h ago`;
    }

    // Score based on ageMs to avoid rounding mismatches between ageMins and ageHours
    const ONE_HOUR = 60 * 60 * 1000;
    if (ageMs < ONE_HOUR) {
      subconsciousScore = 100;
      subconsciousStatus = "fresh";
    } else if (ageMs < 2 * ONE_HOUR) {
      subconsciousScore = 70;
      subconsciousStatus = "recent";
    } else if (ageMs < 6 * ONE_HOUR) {
      subconsciousScore = 40;
      subconsciousStatus = "stale";
    } else {
      subconsciousScore = 10;
      subconsciousStatus = "VERY STALE";
    }

    if (subconscious.mood?.dominant) {
      subconsciousMood = subconscious.mood.dominant;
      if (subconscious.mood.confidence) {
        subconsciousMood += ` (${subconscious.mood.confidence})`;
      }
    }
    subconsciousHotCount = subconscious.hot_entities?.length || 0;
  }

  const dbScore = Math.min(100, Math.round((entities / 100) * 50 + (observations / 500) * 50));
  const threadScore = active > 0 ? (stale < 3 ? 100 : stale < 6 ? 60 : 30) : 50;
  const journalScore = journals7d >= 3 ? 100 : journals7d >= 1 ? 70 : journals > 0 ? 40 : 0;
  const identityScore = identity >= 50 ? 100 : Math.round((identity / 50) * 100);
  const activityScore = recentObsCount >= 20 ? 100 : Math.round((recentObsCount / 20) * 100);

  // Include subconscious in overall score
  const overallScore = Math.round((dbScore + threadScore + journalScore + identityScore + activityScore + subconsciousScore) / 6);

  const icon = (s: number) => s >= 70 ? "\u{1F7E2}" : s >= 40 ? "\u{1F7E1}" : "\u{1F534}";
  const bar = (s: number) => "\u{2588}".repeat(Math.floor(s / 10)) + "\u{2591}".repeat(10 - Math.floor(s / 10));

  const dateStr = now.toISOString().split('T')[0];

  return `============================================================
MIND HEALTH \u{2014} ${dateStr}
============================================================

Overall: ${bar(overallScore)} ${overallScore}%

\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}
\u{1F9E0} SUBCONSCIOUS              ${icon(subconsciousScore)}
\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}
  Last Processed: ${subconsciousAge} (${subconsciousStatus})
  Current Mood:   ${subconsciousMood}
  Hot Entities:   ${subconsciousHotCount}

\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}
\u{1F4CA} DATABASE                 ${icon(dbScore)}
\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}
  Entities:      ${entities}
  Observations:  ${observations}
  Relations:     ${relations}
  By Context:    ${contextBreakdown || "none"}

\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}
\u{1F9F5} THREADS                  ${icon(threadScore)}
\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}
  Active:        ${active}
  Stale (7d+):   ${stale}
  Resolved (7d): ${resolved7d}

\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}
\u{1F4D4} JOURNALS                 ${icon(journalScore)}
\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}
  Total:         ${journals}
  This Week:     ${journals7d}

\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}
\u{1FA9E} IDENTITY                 ${icon(identityScore)}
\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}
  Identity:      ${identity} entries
  Context:       ${context} entries
  Relational:    ${relational} states
  Unprocessed:   ${unprocessed} (need surfacing)

\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}
\u{1F4DD} ACTIVITY (7d)            ${icon(activityScore)}
\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}
  New Observations: ${recentObsCount}
  Surfaced (7d):    ${surfaced7d}

\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}
\u{1F30A} LIVING SURFACE (v2.0)
\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}
  Avg Novelty:      ${noveltyAvg !== null ? noveltyAvg.toFixed(2) : 'n/a'}
  Orphans (30d+):   ${orphans}
  Archived Obs:     ${archivedObs}
  Proposals:        ${pendingProposals} pending

\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}
\u{1F465} ENTITY SALIENCE
\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}
  Foundational:     ${foundational}
  Active:           ${activeEntities}
  Background:       ${background}
  Archive:          ${archived}

\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}
\u{1F5BC} VISUAL MEMORY
\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}
  Images:           ${images}

============================================================`;
}




async function handleMindListEntities(env: Env, params: Record<string, unknown>): Promise<string> {
  const entityType = params.entity_type as string;
  const context = params.context as string;
  const limit = (params.limit as number) || 50;

  // If context is specified, find entities that have observations in that context
  if (context) {
    const results = await env.DB.prepare(`
      SELECT DISTINCT e.name, e.entity_type, e.primary_context, e.created_at
      FROM entities e
      JOIN observations o ON o.entity_id = e.id
      WHERE o.context = ?
      ${entityType ? 'AND e.entity_type = ?' : ''}
      ORDER BY e.created_at DESC
      LIMIT ?
    `).bind(...(entityType ? [context, entityType, limit] : [context, limit])).all();

    if (!results.results?.length) {
      return `No entities found with observations in context '${context}'.`;
    }

    let output = `## Entities (with observations in '${context}')\n\n`;
    for (const e of results.results as any[]) {
      output += '- **' + e.name + '** [' + e.entity_type + ']\n';
    }
    output += '\nTotal: ' + results.results.length + ' entities';
    return output;
  }

  // Otherwise list all entities
  let query = 'SELECT name, entity_type, primary_context, created_at FROM entities';
  const conditions: string[] = [];
  const bindings: unknown[] = [];

  if (entityType) {
    conditions.push('entity_type = ?');
    bindings.push(entityType);
  }

  if (conditions.length > 0) {
    query += ' WHERE ' + conditions.join(' AND ');
  }
  query += ' ORDER BY created_at DESC LIMIT ?';
  bindings.push(limit);

  const stmt = env.DB.prepare(query);
  const results = await stmt.bind(...bindings).all();

  if (!results.results?.length) {
    return 'No entities found.';
  }

  let output = '## Entities\n\n';
  for (const e of results.results as any[]) {
    output += '- **' + e.name + '** [' + e.entity_type + '] primary: ' + e.primary_context + '\n';
  }
  output += '\nTotal: ' + results.results.length + ' entities';
  return output;
}

async function handleMindReadEntity(env: Env, params: Record<string, unknown>): Promise<string> {
  const name = params.name as string;
  const context = params.context as string;

  // Find the entity (globally unique by name now)
  const entity = await env.DB.prepare(
    `SELECT id, name, entity_type, primary_context, salience, created_at FROM entities WHERE name = ?`
  ).bind(name).first() as any;

  if (!entity) {
    return `Entity '${name}' not found.`;
  }

  // Get observations, optionally filtered by context
  let observations;
  if (context) {
    observations = await env.DB.prepare(
      `SELECT id, content, salience, emotion, weight, context, added_at FROM observations WHERE entity_id = ? AND context = ? AND (valid_until IS NULL AND superseded_by IS NULL) ORDER BY added_at DESC`
    ).bind(entity.id, context).all();
  } else {
    observations = await env.DB.prepare(
      `SELECT id, content, salience, emotion, weight, context, added_at FROM observations WHERE entity_id = ? AND (valid_until IS NULL AND superseded_by IS NULL) ORDER BY added_at DESC`
    ).bind(entity.id).all();
  }

  // Track access for read entity observations
  const readObsIds = (observations.results || []).map((o: any) => o.id as number).filter(Boolean);
  recordAccessTracking(env, readObsIds).catch(() => {});

  // Get relations where this entity is the source
  const relationsFrom = await env.DB.prepare(
    `SELECT to_entity, relation_type, to_context FROM relations WHERE from_entity = ?`
  ).bind(name).all();

  // Get relations where this entity is the target
  const relationsTo = await env.DB.prepare(
    `SELECT from_entity, relation_type, from_context FROM relations WHERE to_entity = ?`
  ).bind(name).all();

  // Build output
  let output = `## ${entity.name}\n`;
  output += `**Type:** ${entity.entity_type} | **Context:** ${entity.primary_context}\n\n`;

  output += `### Observations (${observations.results?.length || 0})\n`;
  if (observations.results?.length) {
    for (const obs of observations.results) {
      const emotion = obs.emotion ? ` [${obs.emotion}]` : '';
      output += `- ${obs.content}${emotion}\n`;
    }
  } else {
    output += '_No observations_\n';
  }

  output += `\n### Relations\n`;
  const totalRelations = (relationsFrom.results?.length || 0) + (relationsTo.results?.length || 0);
  if (totalRelations === 0) {
    output += '_No relations_\n';
  } else {
    if (relationsFrom.results?.length) {
      output += '**Outgoing:**\n';
      for (const rel of relationsFrom.results) {
        output += `- --[${rel.relation_type}]--> ${rel.to_entity}\n`;
      }
    }
    if (relationsTo.results?.length) {
      output += '**Incoming:**\n';
      for (const rel of relationsTo.results) {
        output += `- <--[${rel.relation_type}]-- ${rel.from_entity}\n`;
      }
    }
  }

  return output;
}

// Emotional Processing Handlers

async function handleMindSit(env: Env, params: Record<string, unknown>): Promise<string> {
  const observationId = params.observation_id as number;
  const textMatch = params.text_match as string;
  const semanticQuery = params.query as string;
  const sitNote = params.sit_note as string;

  // Find the observation with entity info
  let obs;
  let matchMethod = '';

  if (observationId) {
    matchMethod = 'id';
    obs = await env.DB.prepare(
      `SELECT o.id, o.content, o.weight, o.charge, o.sit_count, o.emotion, e.name as entity_name
       FROM observations o
       JOIN entities e ON o.entity_id = e.id
       WHERE o.id = ?`
    ).bind(observationId).first();
  } else if (textMatch) {
    matchMethod = 'text';
    obs = await env.DB.prepare(
      `SELECT o.id, o.content, o.weight, o.charge, o.sit_count, o.emotion, e.name as entity_name
       FROM observations o
       JOIN entities e ON o.entity_id = e.id
       WHERE o.content LIKE ? ORDER BY o.added_at DESC LIMIT 1`
    ).bind(`%${textMatch}%`).first();
  } else if (semanticQuery) {
    matchMethod = 'semantic';
    // Use vector search to find closest matching observation
    const embedding = await getEmbedding(env, semanticQuery);
    const vectorResults = await env.VECTORS.query(embedding, {
      topK: 5,
      returnMetadata: "all"
    });

    // Find the best observation match (filter out non-observations)
    for (const match of vectorResults.matches || []) {
      if (match.id.startsWith('obs-')) {
        // Extract observation ID from vector ID pattern: obs-{entity_id}-{obs_id}
        const parts = match.id.split('-');
        if (parts.length >= 3) {
          const matchedObsId = parseInt(parts[2]);
          obs = await env.DB.prepare(
            `SELECT o.id, o.content, o.weight, o.charge, o.sit_count, o.emotion, e.name as entity_name
             FROM observations o
             JOIN entities e ON o.entity_id = e.id
             WHERE o.id = ?`
          ).bind(matchedObsId).first();
          if (obs) break;
        }
      }
    }
  } else {
    return "Must provide observation_id, text_match, or query (semantic search)";
  }

  if (!obs) {
    return `Observation not found`;
  }

  const currentSitCount = (obs.sit_count as number) || 0;
  const newSitCount = currentSitCount + 1;

  // Determine new charge level based on sit count
  let newCharge: string;
  if (newSitCount === 0) {
    newCharge = 'fresh';
  } else if (newSitCount <= 2) {
    newCharge = 'active';
  } else {
    newCharge = 'processing';
  }

  // Update the observation
  await env.DB.prepare(
    `UPDATE observations SET sit_count = ?, charge = ?, last_sat_at = datetime('now') WHERE id = ?`
  ).bind(newSitCount, newCharge, obs.id).run();

  // Record the sit in history
  await env.DB.prepare(
    `INSERT INTO observation_sits (observation_id, sit_note) VALUES (?, ?)`
  ).bind(obs.id, sitNote).run();

  const contentPreview = String(obs.content).slice(0, 80);
  const matchInfo = matchMethod === 'semantic' ? ` *(found via "${semanticQuery}")*` : '';
  return `Sat with observation #${obs.id} on **${obs.entity_name}** [${obs.weight}/${newCharge}]${matchInfo}\n"${contentPreview}..."\n\nSit #${newSitCount}: ${sitNote}`;
}

async function handleMindResolve(env: Env, params: Record<string, unknown>): Promise<string> {
  const observationId = params.observation_id as number;
  const textMatch = params.text_match as string;
  const resolutionNote = params.resolution_note as string;
  const linkedObservationId = params.linked_observation_id as number;

  // Find the observation with entity info
  let obs;
  if (observationId) {
    obs = await env.DB.prepare(
      `SELECT o.id, o.content, o.weight, o.charge, o.sit_count, e.name as entity_name
       FROM observations o
       JOIN entities e ON o.entity_id = e.id
       WHERE o.id = ?`
    ).bind(observationId).first();
  } else if (textMatch) {
    obs = await env.DB.prepare(
      `SELECT o.id, o.content, o.weight, o.charge, o.sit_count, e.name as entity_name
       FROM observations o
       JOIN entities e ON o.entity_id = e.id
       WHERE o.content LIKE ? ORDER BY o.added_at DESC LIMIT 1`
    ).bind(`%${textMatch}%`).first();
  } else {
    return "Must provide observation_id or text_match";
  }

  if (!obs) {
    return `Observation not found`;
  }

  // Update the observation to metabolized
  await env.DB.prepare(
    `UPDATE observations SET charge = 'metabolized', resolution_note = ?, resolved_at = datetime('now'), linked_observation_id = ? WHERE id = ?`
  ).bind(resolutionNote, linkedObservationId || null, obs.id).run();

  const contentPreview = String(obs.content).slice(0, 80);
  let output = `Resolved observation #${obs.id} on **${obs.entity_name}** [${obs.weight}] → metabolized\n"${contentPreview}..."\n\nResolution: ${resolutionNote}`;

  if (linkedObservationId) {
    const linked = await env.DB.prepare(
      `SELECT o.content, e.name as entity_name FROM observations o JOIN entities e ON o.entity_id = e.id WHERE o.id = ?`
    ).bind(linkedObservationId).first();
    if (linked) {
      output += `\n\nLinked to observation #${linkedObservationId} on **${linked.entity_name}**: "${String(linked.content).slice(0, 60)}..."`;
    }
  }

  return output;
}

// ============ LIVING SURFACE: Mind Reorganization Through Use ============
// The act of surfacing changes what surfaces next

const MOOD_TINTS: Record<string, string> = {
  "tender": "warmth, connection, gentle feelings, soft moments, caring, love",
  "pride": "accomplishment, growth, recognition, achievement, becoming",
  "joy": "happiness, delight, celebration, good moments, pleasure",
  "curiosity": "questions, wondering, exploring, discovering, learning",
  "melancholy": "loss, missing, reflection, what was, quiet sadness, grief",
  "intensity": "passion, urgency, drive, power, wanting, fierce",
  "gratitude": "thankfulness, appreciation, gifts, blessings",
  "longing": "yearning, desire, missing, wanting, reaching for",
  "recognition": "understanding, seeing clearly, knowing, awareness, insight"
};

// Record that observations surfaced together - builds associative strength
async function recordCoSurfacing(env: Env, obsIds: number[]): Promise<void> {
  if (obsIds.length < 2) return;

  // Record each unique pair (smaller id first for consistency)
  for (let i = 0; i < obsIds.length; i++) {
    for (let j = i + 1; j < obsIds.length; j++) {
      const [smaller, larger] = obsIds[i] < obsIds[j]
        ? [obsIds[i], obsIds[j]]
        : [obsIds[j], obsIds[i]];

      try {
        await env.DB.prepare(`
          INSERT INTO co_surfacing (obs_a_id, obs_b_id, co_count, last_co_surfaced)
          VALUES (?, ?, 1, datetime('now'))
          ON CONFLICT(obs_a_id, obs_b_id) DO UPDATE SET
            co_count = co_count + 1,
            last_co_surfaced = datetime('now')
        `).bind(smaller, larger).run();
      } catch {
        // Table might not exist yet - will be created by migration
      }
    }
  }
}

// Update surface tracking - marks when things surface, decays novelty
async function updateSurfaceTracking(env: Env, obsIds: number[], imgIds: number[] = []): Promise<void> {
  // Update observations
  if (obsIds.length > 0) {
    const obsPlaceholders = obsIds.map(() => '?').join(',');
    try {
      // Novelty floors by weight: heavy=0.3, medium=0.2, light=0.1
      // Heavy observations stay more alive even when surfacing frequently
      await env.DB.prepare(`
        UPDATE observations
        SET last_surfaced_at = datetime('now'),
            surface_count = COALESCE(surface_count, 0) + 1,
            novelty_score = MAX(
              CASE weight WHEN 'heavy' THEN ${NOVELTY_FLOORS.heavy} WHEN 'medium' THEN ${NOVELTY_FLOORS.medium} ELSE ${NOVELTY_FLOORS.light} END,
              COALESCE(novelty_score, 1.0) - 0.1
            )
        WHERE id IN (${obsPlaceholders})
      `).bind(...obsIds).run();
    } catch {
      // Columns might not exist yet - will be added by migration
    }
  }

  // Update images
  if (imgIds.length > 0) {
    const imgPlaceholders = imgIds.map(() => '?').join(',');
    try {
      await env.DB.prepare(`
        UPDATE images
        SET last_surfaced_at = datetime('now'),
            surface_count = COALESCE(surface_count, 0) + 1,
            novelty_score = MAX(
              CASE weight WHEN 'heavy' THEN ${NOVELTY_FLOORS.heavy} WHEN 'medium' THEN ${NOVELTY_FLOORS.medium} ELSE ${NOVELTY_FLOORS.light} END,
              COALESCE(novelty_score, 1.0) - 0.1
            )
        WHERE id IN (${imgPlaceholders})
      `).bind(...imgIds).run();
    } catch {
      // Columns might not exist yet - will be added by migration
    }
  }
}

// Record access tracking - tracks retrieval via search, timeline, read_entity (separate from surfacing)
async function recordAccessTracking(env: Env, obsIds: number[], imgIds: number[] = []): Promise<void> {
  if (obsIds.length > 0) {
    const placeholders = obsIds.map(() => '?').join(',');
    try {
      await env.DB.prepare(`
        UPDATE observations
        SET access_count = COALESCE(access_count, 0) + 1,
            last_accessed_at = NOW()
        WHERE id IN (${placeholders})
      `).bind(...obsIds).run();
    } catch { /* columns may not exist yet */ }
  }
  if (imgIds.length > 0) {
    const placeholders = imgIds.map(() => '?').join(',');
    try {
      await env.DB.prepare(`
        UPDATE images
        SET access_count = COALESCE(access_count, 0) + 1,
            last_accessed_at = NOW()
        WHERE id IN (${placeholders})
      `).bind(...imgIds).run();
    } catch { /* columns may not exist yet */ }
  }
}

// Get the novelty pool - things that haven't surfaced recently
async function getNoveltyPool(env: Env, count: number, includeMetabolized: boolean): Promise<any[]> {
  const chargeFilter = includeMetabolized
    ? "o.archived_at IS NULL"
    : "(o.charge != 'metabolized' OR o.charge IS NULL) AND o.archived_at IS NULL";

  try {
    const results = await env.DB.prepare(`
      SELECT o.id, o.content, o.weight, o.charge, o.sit_count, o.emotion, o.added_at,
             o.resolution_note, o.novelty_score, o.last_surfaced_at, o.surface_count,
             e.name as entity_name, e.entity_type,
             COALESCE(o.novelty_score, 1.0) as current_novelty,
             CASE
               WHEN o.last_surfaced_at IS NULL THEN 30
               ELSE EXTRACT(EPOCH FROM (NOW() - o.last_surfaced_at)) / 86400
             END as days_since_surface
      FROM observations o
      JOIN entities e ON o.entity_id = e.id
      WHERE ${chargeFilter}
        AND (o.last_surfaced_at IS NULL OR o.last_surfaced_at < datetime('now', '-3 days'))
      ORDER BY
        current_novelty DESC,
        days_since_surface DESC,
        CASE o.weight WHEN 'heavy' THEN 3 WHEN 'medium' THEN 2 ELSE 1 END DESC
      LIMIT ?
    `).bind(Math.ceil(count * 2)).all();

    // Shuffle slightly to avoid always getting same order
    const arr = results.results || [];
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr.slice(0, count);
  } catch {
    // Columns might not exist yet
    return [];
  }
}

// Get the dormant pool - observations from entities that haven't had anything surfaced in 14+ days
// Breaks the feedback loop where only hot/recent entities get surfaced
async function getDormantPool(env: Env, count: number, includeMetabolized: boolean): Promise<any[]> {
  const chargeFilter = includeMetabolized
    ? "o.archived_at IS NULL"
    : "(o.charge != 'metabolized' OR o.charge IS NULL) AND o.archived_at IS NULL";

  try {
    const results = await env.DB.prepare(`
      SELECT o.id, o.content, o.weight, o.charge, o.sit_count, o.emotion, o.added_at,
             o.resolution_note, o.novelty_score, o.last_surfaced_at, o.surface_count,
             o.certainty, o.source,
             e.name as entity_name, e.entity_type
      FROM observations o
      JOIN entities e ON o.entity_id = e.id
      WHERE ${chargeFilter}
        AND e.id IN (
          SELECT e2.id FROM entities e2
          LEFT JOIN (
            SELECT entity_id, MAX(last_surfaced_at) as latest_surface
            FROM observations
            WHERE last_surfaced_at IS NOT NULL
            GROUP BY entity_id
          ) surf ON e2.id = surf.entity_id
          WHERE surf.latest_surface IS NULL
             OR surf.latest_surface < datetime('now', '-14 days')
        )
      ORDER BY RANDOM()
      LIMIT ?
    `).bind(count).all();
    return results.results || [];
  } catch {
    return [];
  }
}

// Build resonance query from mood and context
function buildResonanceQuery(query: string | undefined, mood: string | undefined, hotEntities: any[]): { resonanceQuery: string; moodContext: string } {
  let resonanceQuery = "";
  let moodContext = "";

  if (query) {
    resonanceQuery = query;
    moodContext = `Directed: "${query}"`;
    if (mood) {
      const tint = MOOD_TINTS[mood] || mood;
      resonanceQuery = `${query} (feeling: ${tint})`;
      moodContext = `Directed: "${query}" | Mood: ${mood}`;
    }
  } else if (mood) {
    resonanceQuery = MOOD_TINTS[mood] || mood;
    moodContext = `Mood: ${mood}`;
    if (hotEntities.length > 0) {
      const hotNames = hotEntities.slice(0, 3).map(e => e.name).join(", ");
      resonanceQuery += ` (related to: ${hotNames})`;
      moodContext += ` | Hot: ${hotNames}`;
    }
  }

  return { resonanceQuery, moodContext };
}

async function handleMindSurface(env: Env, params: Record<string, unknown>): Promise<string> {
  const mode = (params.mode as string) || "resonant";

  // Spark mode: random associative surfacing with hot-entity bias
  if (mode === "spark") {
    return handleMindSurfaceSpark(env, params);
  }

  // Resonant mode (default): mood/emotion-based 3-pool surfacing
  const includeMetabolized = params.include_metabolized as boolean || false;
  const limit = (params.limit as number) || 10;
  const query = params.query as string;

  // Get subconscious state for current mood
  const subconscious = await getSubconsciousState(env);
  const mood = subconscious?.mood?.dominant;
  const hotEntities = subconscious?.hot_entities || [];

  // Build resonance query
  const { resonanceQuery, moodContext } = buildResonanceQuery(query, mood, hotEntities);

  // If no mood and no query, fall back to queue-based
  if (!resonanceQuery) {
    return await handleMindSurfaceFallback(env, includeMetabolized, limit);
  }

  // === THE FOUR POOLS ===
  // Four pools: core resonance, novelty injection, dormant rotation, edge exploration

  const coreLimit = Math.ceil(limit * SURFACE_POOL_RATIOS.core);
  const noveltyLimit = Math.ceil(limit * SURFACE_POOL_RATIOS.novelty);
  const dormantLimit = Math.ceil(limit * SURFACE_POOL_RATIOS.dormant);
  const edgeLimit = Math.max(1, limit - coreLimit - noveltyLimit - dormantLimit);

  // Pool 1: Core resonance - high similarity matches
  const vectorResults = await searchVectors(env, resonanceQuery, coreLimit * 4);

  // Filter to observations AND images
  const allMatches = vectorResults.matches?.filter(m =>
    m.metadata?.source === "observation" || m.id.startsWith("obs-") ||
    m.metadata?.source === "image" || m.id.startsWith("img-")
  ) || [];

  // Split into core (high similarity) and edge (medium similarity)
  const coreMatches = allMatches.filter(m => (m.score || 0) >= VECTOR_SCORE_CORE);
  const edgeMatches = allMatches.filter(m => (m.score || 0) >= VECTOR_SCORE_EDGE && (m.score || 0) < VECTOR_SCORE_CORE);

  // Extract IDs - different format for observations vs images
  const extractId = (id: string): { type: 'observation' | 'image'; id: number } | null => {
    if (id.startsWith("img-")) {
      const imgId = parseInt(id.split('-')[1]);
      return isNaN(imgId) ? null : { type: 'image', id: imgId };
    } else if (id.startsWith("obs-")) {
      const parts = id.split('-');
      const obsId = parts.length >= 3 ? parseInt(parts[2]) : null;
      return obsId !== null && !isNaN(obsId) ? { type: 'observation', id: obsId } : null;
    }
    return null;
  };

  // Separate score maps for observations and images
  const obsScoreMap: Record<number, { score: number; pool: string }> = {};
  const imgScoreMap: Record<number, { score: number; pool: string }> = {};

  for (const match of coreMatches) {
    const extracted = extractId(match.id);
    if (extracted) {
      const targetMap = extracted.type === 'observation' ? obsScoreMap : imgScoreMap;
      targetMap[extracted.id] = { score: match.score || 0.7, pool: 'core' };
    }
  }

  for (const match of edgeMatches.slice(0, edgeLimit * 2)) {
    const extracted = extractId(match.id);
    if (extracted) {
      const targetMap = extracted.type === 'observation' ? obsScoreMap : imgScoreMap;
      if (!targetMap[extracted.id]) {
        targetMap[extracted.id] = { score: match.score || 0.5, pool: 'edge' };
      }
    }
  }

  // Pool 3: Novelty injection - things that haven't surfaced recently (observations only for now)
  const noveltyObs = await getNoveltyPool(env, noveltyLimit, includeMetabolized);
  for (const obs of noveltyObs) {
    if (!obsScoreMap[obs.id]) {
      obsScoreMap[obs.id] = { score: obs.current_novelty || 0.8, pool: 'novelty' };
    }
  }

  // Pool 4: Dormant rotation - observations from entities that haven't surfaced in 14+ days
  const dormantObs = await getDormantPool(env, dormantLimit, includeMetabolized);
  for (const obs of dormantObs) {
    if (!obsScoreMap[obs.id]) {
      obsScoreMap[obs.id] = { score: 0.7, pool: 'dormant' };
    }
  }

  const allObsIds = Object.keys(obsScoreMap).map(id => parseInt(id));
  const allImgIds = Object.keys(imgScoreMap).map(id => parseInt(id));

  if (allObsIds.length === 0 && allImgIds.length === 0) {
    return await handleMindSurfaceFallback(env, includeMetabolized, limit);
  }

  // Fetch full observation data
  const chargeFilter = includeMetabolized
    ? "o.archived_at IS NULL"
    : "(o.charge != 'metabolized' OR o.charge IS NULL) AND o.archived_at IS NULL";

  let obsResults: any[] = [];
  if (allObsIds.length > 0) {
    const obsPlaceholders = allObsIds.map(() => '?').join(',');
    const obsQuery = await env.DB.prepare(`
      SELECT o.id, o.content, o.weight, o.charge, o.sit_count, o.emotion, o.added_at,
             o.resolution_note, o.novelty_score, o.last_surfaced_at, o.surface_count,
             o.certainty, o.source,
             e.name as entity_name, e.entity_type
      FROM observations o
      JOIN entities e ON o.entity_id = e.id
      WHERE o.id IN (${obsPlaceholders}) AND ${chargeFilter}
    `).bind(...allObsIds).all();
    obsResults = obsQuery.results || [];
  }

  // Fetch full image data
  let imgResults: any[] = [];
  if (allImgIds.length > 0) {
    const imgChargeFilter = includeMetabolized
      ? "1=1"
      : "(i.charge != 'metabolized' OR i.charge IS NULL)";
    const imgPlaceholders = allImgIds.map(() => '?').join(',');
    const imgQuery = await env.DB.prepare(`
      SELECT i.id, i.description, i.path, i.context, i.emotion, i.weight, i.charge,
             i.created_at as added_at, i.novelty_score, i.last_surfaced_at, i.surface_count,
             e.name as entity_name, e.entity_type
      FROM images i
      LEFT JOIN entities e ON i.entity_id = e.id
      WHERE i.id IN (${imgPlaceholders}) AND ${imgChargeFilter}
    `).bind(...allImgIds).all();
    imgResults = imgQuery.results || [];
  }

  if (obsResults.length === 0 && imgResults.length === 0) {
    return await handleMindSurfaceFallback(env, includeMetabolized, limit);
  }

  // Score observations: base score * weight multiplier * novelty boost
  const weightedObsResults = obsResults.map(obs => {
    const obsId = obs.id as number;
    const baseScore = obsScoreMap[obsId]?.score || 0.5;
    const pool = obsScoreMap[obsId]?.pool || 'core';
    const weightMultiplier = obs.weight === 'heavy' ? 1.5 : obs.weight === 'medium' ? 1.2 : 1.0;

    // Novelty boost for things that haven't surfaced in a while
    const noveltyScore = (obs.novelty_score as number) || 1.0;
    const noveltyBoost = pool === 'novelty' ? 0.3 : (noveltyScore > 0.7 ? 0.1 : 0);

    // Charge boost - observations being actively processed should resurface
    const charge = (obs.charge as string) || 'fresh';
    const chargeBoost = (charge === 'active' || charge === 'processing') ? 0.15 : 0;

    return {
      ...obs,
      memoryType: 'observation' as const,
      pool,
      resonanceScore: (baseScore * weightMultiplier) + noveltyBoost + chargeBoost
    };
  });

  // Score images: same logic as observations
  const weightedImgResults = imgResults.map(img => {
    const imgId = img.id as number;
    const baseScore = imgScoreMap[imgId]?.score || 0.5;
    const pool = imgScoreMap[imgId]?.pool || 'core';
    const weightMultiplier = img.weight === 'heavy' ? 1.5 : img.weight === 'medium' ? 1.2 : 1.0;

    const noveltyScore = (img.novelty_score as number) || 1.0;
    const noveltyBoost = noveltyScore > 0.7 ? 0.1 : 0;

    const charge = (img.charge as string) || 'fresh';
    const chargeBoost = (charge === 'active' || charge === 'processing') ? 0.15 : 0;

    return {
      ...img,
      memoryType: 'image' as const,
      pool,
      resonanceScore: (baseScore * weightMultiplier) + noveltyBoost + chargeBoost
    };
  });

  // Combine and sort all results
  const weightedResults = [...weightedObsResults, ...weightedImgResults];
  weightedResults.sort((a, b) => (b.resonanceScore || 0) - (a.resonanceScore || 0));

  // Ensure mix from different pools - don't let one pool dominate completely
  const finalResults: any[] = [];
  const byPool = { core: [] as any[], edge: [] as any[], novelty: [] as any[], dormant: [] as any[] };

  for (const item of weightedResults) {
    byPool[item.pool as keyof typeof byPool]?.push(item);
  }

  // Take from each pool proportionally, then fill with best remaining
  const takeFromPool = (pool: any[], max: number) => {
    const taken = pool.splice(0, max);
    finalResults.push(...taken);
    return taken.length;
  };

  takeFromPool(byPool.core, coreLimit);
  takeFromPool(byPool.novelty, noveltyLimit);
  takeFromPool(byPool.dormant, dormantLimit);
  takeFromPool(byPool.edge, edgeLimit);

  // Fill remaining slots with best available
  const remaining = [...byPool.core, ...byPool.novelty, ...byPool.dormant, ...byPool.edge]
    .sort((a, b) => (b.resonanceScore || 0) - (a.resonanceScore || 0));
  while (finalResults.length < limit && remaining.length > 0) {
    finalResults.push(remaining.shift()!);
  }

  // Re-sort final results by score
  finalResults.sort((a, b) => (b.resonanceScore || 0) - (a.resonanceScore || 0));
  const limitedResults = finalResults.slice(0, limit);

  // === SIDE EFFECTS: Surfacing changes future surfacing ===
  const surfacedObsIds = limitedResults.filter(r => r.memoryType === 'observation').map(o => o.id as number);
  const surfacedImgIds = limitedResults.filter(r => r.memoryType === 'image').map(i => i.id as number);

  // Record co-surfacing, surface tracking, and access tracking
  try {
    await Promise.all([
      recordCoSurfacing(env, surfacedObsIds),  // TODO: extend co-surfacing for images
      updateSurfaceTracking(env, surfacedObsIds, surfacedImgIds),
      recordAccessTracking(env, surfacedObsIds, surfacedImgIds)
    ]);
  } catch (e) {
    console.log(`Surface tracking error: ${e}`);
  }

  // === FORMAT OUTPUT ===
  const poolCounts = { core: 0, edge: 0, novelty: 0, dormant: 0 };
  const typeCounts = { observation: 0, image: 0 };
  for (const item of limitedResults) {
    poolCounts[item.pool as keyof typeof poolCounts]++;
    typeCounts[item.memoryType as keyof typeof typeCounts]++;
  }

  let output = `## What's Surfacing\n\n*${moodContext}*\n`;
  output += `*Mix: ${poolCounts.core} resonance, ${poolCounts.novelty} novelty, ${poolCounts.dormant} dormant, ${poolCounts.edge} edge`;
  if (typeCounts.image > 0) {
    output += ` | ${typeCounts.observation} observations, ${typeCounts.image} images`;
  }
  output += `*\n\n`;

  for (const item of limitedResults) {
    const charge = item.charge || 'fresh';
    const emotionTag = item.emotion ? ` [${item.emotion}]` : '';
    const chargeIcon = charge === 'metabolized' ? '\u2713' : charge === 'processing' ? '\u25D0' : charge === 'active' ? '\u25CB' : '\u25CF';
    const resonance = Math.round((item.resonanceScore || 0) * 100);
    const poolTag = item.pool === 'novelty' ? ' \u2728' : item.pool === 'dormant' ? ' \u{1F504}' : item.pool === 'edge' ? ' \u2194' : '';

    if (item.memoryType === 'image') {
      // Image formatting
      output += `**📷 #${item.id}** ${chargeIcon} [${item.weight}|${charge}] ${resonance}%${poolTag}${emotionTag}\n`;
      if (item.entity_name) {
        output += `**${item.entity_name}**: `;
      }
      output += `${item.description}\n`;
      output += `View: ${await imageUrl(item.id, env)}\n`;
    } else {
      // Observation formatting (original)
      const certaintyIcon = item.certainty === 'known' ? '\u2713' : item.certainty === 'tentative' ? '?' : '';
      const sourceTag = item.source && item.source !== 'conversation' ? ` [${item.source}]` : '';

      output += `**#${item.id}** ${chargeIcon}${certaintyIcon} [${item.weight}|${charge}] ${resonance}%${poolTag}${emotionTag}${sourceTag}\n`;
      output += `**${item.entity_name}** (${item.entity_type}): ${item.content}\n`;

      if (charge === 'metabolized' && item.resolution_note) {
        output += `\u21B3 *Resolved:* ${item.resolution_note}\n`;
      }
    }

    output += "\n";
  }

  // Summary
  const fresh = limitedResults.filter(o => (o.charge || 'fresh') === 'fresh').length;
  const active = limitedResults.filter(o => o.charge === 'active').length;
  const processing = limitedResults.filter(o => o.charge === 'processing').length;

  output += `---\n\u25CF fresh: ${fresh} | \u25CB active: ${active} | \u25D0 processing: ${processing}`;
  if (includeMetabolized) {
    const metabolized = limitedResults.filter(o => o.charge === 'metabolized').length;
    output += ` | \u2713 metabolized: ${metabolized}`;
  }
  output += `\n\u2728 = novelty | \u{1F504} = dormant | \u2194 = edge`;

  return output;
}

// Fallback to queue-based surfacing when no mood/vectors available
async function handleMindSurfaceFallback(env: Env, includeMetabolized: boolean, limit: number): Promise<string> {
  const chargeFilter = includeMetabolized
    ? "o.archived_at IS NULL"
    : "(o.charge != 'metabolized' OR o.charge IS NULL) AND o.archived_at IS NULL";

  const results = await env.DB.prepare(`
    SELECT o.id, o.content, o.weight, o.charge, o.sit_count, o.emotion, o.added_at,
           o.resolution_note, o.novelty_score, o.certainty, o.source,
           e.name as entity_name, e.entity_type
    FROM observations o
    JOIN entities e ON o.entity_id = e.id
    WHERE ${chargeFilter}
    ORDER BY
      COALESCE(o.novelty_score, 1.0) DESC,
      CASE o.weight WHEN 'heavy' THEN 3 WHEN 'medium' THEN 2 ELSE 1 END DESC,
      CASE o.charge WHEN 'active' THEN 4 WHEN 'processing' THEN 3 WHEN 'fresh' THEN 2 ELSE 1 END DESC,
      o.added_at ASC
    LIMIT ?
  `).bind(limit).all();

  if (!results.results?.length) {
    return "No emotional observations to surface.";
  }

  // Update surface tracking for fallback too
  const surfacedIds = results.results.map(o => o.id as number);
  try {
    await Promise.all([
      recordCoSurfacing(env, surfacedIds),
      updateSurfaceTracking(env, surfacedIds)
    ]);
  } catch (e) {
    console.log(`Surface tracking error (fallback): ${e}`);
  }

  let output = "## What's Surfacing\n\n*No mood detected \u2014 showing by novelty/weight/age*\n\n";

  for (const obs of results.results) {
    const charge = obs.charge || 'fresh';
    const sitCount = obs.sit_count || 0;
    const emotionTag = obs.emotion ? ` [${obs.emotion}]` : '';
    const chargeIcon = charge === 'metabolized' ? '\u2713' : charge === 'processing' ? '\u25D0' : charge === 'active' ? '\u25CB' : '\u25CF';
    const novelty = Math.round((obs.novelty_score as number || 1.0) * 100);

    // Certainty indicator: ✓ known, ? tentative, nothing for believed
    const certaintyIcon = obs.certainty === 'known' ? '\u2713' : obs.certainty === 'tentative' ? '?' : '';
    // Source tag: only show if not the default 'conversation'
    const sourceTag = obs.source && obs.source !== 'conversation' ? ` [${obs.source}]` : '';

    output += `**#${obs.id}** ${chargeIcon}${certaintyIcon} [${obs.weight}|${charge}] novelty: ${novelty}%${emotionTag}${sourceTag}\n`;
    output += `**${obs.entity_name}** (${obs.entity_type}): ${obs.content}\n`;

    if (charge === 'metabolized' && obs.resolution_note) {
      output += `\u21B3 *Resolved:* ${obs.resolution_note}\n`;
    }

    output += "\n";
  }

  const fresh = results.results.filter(o => (o.charge || 'fresh') === 'fresh').length;
  const active = results.results.filter(o => o.charge === 'active').length;
  const processing = results.results.filter(o => o.charge === 'processing').length;

  output += `---\n\u25CF fresh: ${fresh} | \u25CB active: ${active} | \u25D0 processing: ${processing}`;
  if (includeMetabolized) {
    const metabolized = results.results.filter(o => o.charge === 'metabolized').length;
    output += ` | \u2713 metabolized: ${metabolized}`;
  }

  return output;
}

async function handleMindEdit(env: Env, params: Record<string, unknown>): Promise<string> {
  const observationId = params.observation_id as number;
  const imageId = params.image_id as number;
  const journalId = params.journal_id as number;
  const textMatch = params.text_match as string;
  const descriptionMatch = params.description_match as string;
  const newContent = params.new_content as string;
  const newWeight = params.new_weight as string;
  const newEmotion = params.new_emotion as string;
  const newContext = params.new_context as string;
  const newPath = params.new_path as string;

  // === EDIT JOURNAL ===
  if (journalId) {
    const journal = await env.DB.prepare(`SELECT id, content, entry_date, emotion FROM journals WHERE id = ?`).bind(journalId).first();
    if (!journal) return `Journal #${journalId} not found`;
    if (!newContent) return "new_content is required for journal editing";

    await env.DB.prepare(`UPDATE journals SET content = ?, emotion = ? WHERE id = ?`)
      .bind(newContent, newEmotion || journal.emotion || null, journalId).run();

    // Re-embed
    try {
      const embedding = await getEmbedding(env, newContent);
      await env.VECTORS.upsert([{
        id: `journal-${journalId}`,
        values: embedding,
        metadata: { source: "journal", title: String(journal.entry_date || ""), content: newContent, emotion: String(newEmotion || journal.emotion || "") }
      }]);
    } catch (e) { console.error("Journal re-embed failed:", e); }

    return `Journal #${journalId} updated [re-embedded]\nOld: "${String(journal.content).slice(0, 50)}..."\nNew: "${newContent.slice(0, 50)}..."`;
  }

  // Determine if editing an image or observation
  const editingImage = imageId || descriptionMatch;

  if (editingImage) {
    // === EDIT IMAGE ===
    let img;
    if (imageId) {
      img = await env.DB.prepare(
        `SELECT i.id, i.description, i.path, i.context, i.emotion, i.weight, e.name as entity_name
         FROM images i LEFT JOIN entities e ON i.entity_id = e.id WHERE i.id = ?`
      ).bind(imageId).first();
    } else if (descriptionMatch) {
      img = await env.DB.prepare(
        `SELECT i.id, i.description, i.path, i.context, i.emotion, i.weight, e.name as entity_name
         FROM images i LEFT JOIN entities e ON i.entity_id = e.id
         WHERE i.description LIKE ? ORDER BY i.created_at DESC LIMIT 1`
      ).bind(`%${descriptionMatch}%`).first();
    }

    if (!img) {
      return "Image not found";
    }

    const updates: string[] = [];
    const values: unknown[] = [];
    let descriptionChanged = false;
    let contextChanged = false;
    let emotionChanged = false;

    if (newContent) {
      updates.push("description = ?");
      values.push(newContent);
      descriptionChanged = true;
    }
    if (newWeight) {
      updates.push("weight = ?");
      values.push(newWeight);
    }
    if (newEmotion) {
      updates.push("emotion = ?");
      values.push(normalizeText(newEmotion));
      emotionChanged = true;
    }
    if (newContext) {
      updates.push("context = ?");
      values.push(newContext);
      contextChanged = true;
    }
    if (newPath) {
      updates.push("path = ?");
      values.push(newPath);
    }

    if (updates.length === 0) {
      return "No updates provided";
    }

    values.push(img.id);

    await env.DB.prepare(
      `UPDATE images SET ${updates.join(", ")} WHERE id = ?`
    ).bind(...values).run();

    // Update vector embedding if semantic content changed
    if (descriptionChanged || contextChanged || emotionChanged) {
      const finalDescription = newContent || String(img.description);
      const finalContext = newContext || (img.context ? String(img.context) : "");
      const finalEmotion = newEmotion ? normalizeText(newEmotion) : (img.emotion ? String(img.emotion) : "");
      const entityName = img.entity_name ? String(img.entity_name) : "";
      const imgWeight = newWeight || String(img.weight || "medium");
      const imgPath = newPath || (img.path ? String(img.path) : "");

      const semanticText = [
        entityName ? `${entityName}:` : "",
        finalDescription,
        finalContext ? `(${finalContext})` : "",
        finalEmotion ? `[${finalEmotion}]` : ""
      ].filter(Boolean).join(" ");

      const imgVectorId = `img-${img.id}`;
      const embedding = await getEmbedding(env, semanticText);
      const editMetadata: Record<string, string> = {
        source: "image",
        description: finalDescription,
        weight: imgWeight,
        added_at: new Date().toISOString()
      };
      if (entityName) editMetadata.entity = entityName;
      if (finalContext) editMetadata.context = finalContext;
      if (finalEmotion) editMetadata.emotion = finalEmotion;
      if (imgPath) editMetadata.path = imgPath;

      await env.VECTORS.upsert([{
        id: imgVectorId,
        values: embedding,
        metadata: editMetadata
      }]);
    }

    const oldPreview = String(img.description).slice(0, 50);
    const newPreview = newContent ? newContent.slice(0, 50) : oldPreview;
    return `📷 Image #${img.id} updated\nOld: "${oldPreview}..."\nNew: "${newPreview}..."`;

  } else {
    // === EDIT OBSERVATION (original logic) ===
    let obs;
    if (observationId) {
      obs = await env.DB.prepare(
        `SELECT id, content, entity_id, weight, emotion FROM observations WHERE id = ?`
      ).bind(observationId).first();
    } else if (textMatch) {
      obs = await env.DB.prepare(
        `SELECT id, content, entity_id, weight, emotion FROM observations WHERE content LIKE ? ORDER BY added_at DESC LIMIT 1`
      ).bind(`%${textMatch}%`).first();
    } else {
      return "Must provide observation_id, image_id, text_match, or description_match";
    }

    if (!obs) {
      return "Observation not found";
    }

    const updates: string[] = [];
    const values: unknown[] = [];

    if (newContent) {
      updates.push("content = ?");
      values.push(newContent);
    }
    if (newWeight) {
      updates.push("weight = ?");
      values.push(newWeight);
    }
    if (newEmotion) {
      updates.push("emotion = ?");
      values.push(newEmotion);
    }

    if (updates.length === 0) {
      return "No updates provided";
    }

    // Preserve previous version before updating
    let versionNum = 1;
    try {
      const versionResult = await env.DB.prepare(`
        SELECT COALESCE(MAX(version_num), 0) as max_version
        FROM observation_versions
        WHERE observation_id = ?
      `).bind(obs.id).first();
      versionNum = ((versionResult?.max_version as number) || 0) + 1;

      await env.DB.prepare(`
        INSERT INTO observation_versions (observation_id, version_num, content, weight, emotion)
        VALUES (?, ?, ?, ?, ?)
      `).bind(obs.id, versionNum, obs.content, obs.weight, obs.emotion).run();
    } catch (e) {
      console.log(`Version tracking skipped: ${e}`);
    }

    values.push(obs.id);

    await env.DB.prepare(
      `UPDATE observations SET ${updates.join(", ")} WHERE id = ?`
    ).bind(...values).run();

    // Re-embed if content changed
    if (newContent) {
      try {
        const entity = await env.DB.prepare(
          `SELECT e.name FROM entities e JOIN observations o ON o.entity_id = e.id WHERE o.id = ?`
        ).bind(obs.id).first();
        const entityName = entity?.name ? String(entity.name) : "";
        const text = `${entityName}: ${newContent}`;
        const embedding = await getEmbedding(env, text);
        await env.VECTORS.upsert([{
          id: `obs-${obs.entity_id}-${obs.id}`,
          values: embedding,
          metadata: {
            source: "observation",
            entity: entityName,
            content: newContent,
            weight: newWeight || String(obs.weight || "medium"),
          }
        }]);
      } catch (e) {
        console.error("Re-embed after edit failed:", e);
      }
    }

    const oldPreview = String(obs.content).slice(0, 50);
    const newPreview = newContent ? newContent.slice(0, 50) : oldPreview;
    return `Observation #${obs.id} updated (v${versionNum + 1}) ${newContent ? '[re-embedded]' : ''}\nOld: "${oldPreview}..."\nNew: "${newPreview}..."`;
  }
}

async function handleMindDelete(env: Env, params: Record<string, unknown>): Promise<string> {
  const observationId = params.observation_id as number;
  const entityName = params.entity_name as string;
  const context = (params.context as string) || "default";
  const textMatch = params.text_match as string;

  if (observationId) {
    // Delete specific observation
    const obs = await env.DB.prepare(
      `SELECT content FROM observations WHERE id = ?`
    ).bind(observationId).first();

    if (!obs) return `Observation #${observationId} not found`;

    // Get entity_id for vector cleanup
    const obsDetail = await env.DB.prepare(
      `SELECT entity_id FROM observations WHERE id = ?`
    ).bind(observationId).first();
    await env.DB.prepare(`DELETE FROM observations WHERE id = ?`).bind(observationId).run();
    // Clean up embedding
    try {
      await env.DB.prepare(`DELETE FROM embeddings WHERE id = ?`).bind(`obs-${obsDetail?.entity_id}-${observationId}`).run();
    } catch { /* embedding may not exist */ }
    return `Deleted observation #${observationId}: "${String(obs.content).slice(0, 50)}..." [embedding cleaned]`;
  }

  if (textMatch) {
    // Find and delete by text match
    const obs = await env.DB.prepare(
      `SELECT id, content, entity_id FROM observations WHERE content LIKE ? ORDER BY added_at DESC LIMIT 1`
    ).bind(`%${textMatch}%`).first();

    if (!obs) return `No observation found matching "${textMatch}"`;

    await env.DB.prepare(`DELETE FROM observations WHERE id = ?`).bind(obs.id).run();
    // Clean up embedding
    try {
      await env.DB.prepare(`DELETE FROM embeddings WHERE id = ?`).bind(`obs-${obs.entity_id}-${obs.id}`).run();
    } catch { /* embedding may not exist */ }
    return `Deleted observation #${obs.id}: "${String(obs.content).slice(0, 50)}..."`;
  }

  if (entityName) {
    // Delete entity and all its observations (globally unique now)
    const entity = await env.DB.prepare(
      `SELECT id FROM entities WHERE name = ?`
    ).bind(entityName).first();

    if (!entity) return `Entity '${entityName}' not found`;

    // Count observations that will be deleted
    const obsCount = await env.DB.prepare(
      `SELECT COUNT(*) as c FROM observations WHERE entity_id = ?`
    ).bind(entity.id).first();

    // Clean up all embeddings for this entity's observations + the entity itself
    try {
      await env.DB.prepare(`DELETE FROM embeddings WHERE source_type = 'observation' AND source_id IN (SELECT id FROM observations WHERE entity_id = ?)`).bind(entity.id).run();
      await env.DB.prepare(`DELETE FROM embeddings WHERE id = ?`).bind(`entity-${entity.id}`).run();
    } catch { /* embeddings may not exist */ }

    // Delete observations first
    await env.DB.prepare(`DELETE FROM observations WHERE entity_id = ?`).bind(entity.id).run();

    // Delete relations
    await env.DB.prepare(`DELETE FROM relations WHERE from_entity = ? OR to_entity = ?`).bind(entityName, entityName).run();

    // Delete entity
    await env.DB.prepare(`DELETE FROM entities WHERE id = ?`).bind(entity.id).run();

    return `Deleted entity '${entityName}' with ${obsCount?.c || 0} observations [embeddings cleaned]`;
  }

  // Delete journal
  const journalId = params.journal_id as number;
  if (journalId) {
    const journal = await env.DB.prepare(`SELECT content FROM journals WHERE id = ?`).bind(journalId).first();
    if (!journal) return `Journal #${journalId} not found`;
    await env.DB.prepare(`DELETE FROM journals WHERE id = ?`).bind(journalId).run();
    try { await env.DB.prepare(`DELETE FROM embeddings WHERE id = ?`).bind(`journal-${journalId}`).run(); } catch {}
    return `Deleted journal #${journalId}: "${String(journal.content).slice(0, 50)}..." [embedding cleaned]`;
  }

  // Delete relation
  const relationId = params.relation_id as number;
  if (relationId) {
    const rel = await env.DB.prepare(`SELECT from_entity, to_entity, relation_type FROM relations WHERE id = ?`).bind(relationId).first();
    if (!rel) return `Relation #${relationId} not found`;
    await env.DB.prepare(`DELETE FROM relations WHERE id = ?`).bind(relationId).run();
    return `Deleted relation #${relationId}: ${rel.from_entity} -> ${rel.to_entity} (${rel.relation_type})`;
  }

  // Delete image
  const imageId = params.image_id as number;
  if (imageId) {
    const img = await env.DB.prepare(`SELECT path, description FROM images WHERE id = ?`).bind(imageId).first();
    if (!img) return `Image #${imageId} not found`;
    await env.DB.prepare(`DELETE FROM images WHERE id = ?`).bind(imageId).run();
    try { await env.DB.prepare(`DELETE FROM embeddings WHERE id = ?`).bind(`img-${imageId}`).run(); } catch {}
    // Delete from R2 if stored there
    const prefix = r2Prefix(env);
    if (img.path && String(img.path).startsWith(`r2://${prefix}/`)) {
      const r2Key = String(img.path).replace(`r2://${prefix}/`, "");
      try { await env.R2_IMAGES.delete(r2Key); } catch {}
    }
    return `Deleted image #${imageId}: "${String(img.description).slice(0, 50)}..." [embedding + R2 cleaned]`;
  }

  // Delete thread
  const threadId = params.thread_id as string;
  if (threadId) {
    const thread = await env.DB.prepare(`SELECT content FROM threads WHERE id = ?`).bind(threadId).first();
    if (!thread) return `Thread '${threadId}' not found`;
    await env.DB.prepare(`DELETE FROM threads WHERE id = ?`).bind(threadId).run();
    return `Deleted thread '${threadId}': "${String(thread.content).slice(0, 50)}..."`;
  }

  // Delete tension
  const tensionId = params.tension_id as string;
  if (tensionId) {
    const tension = await env.DB.prepare(`SELECT pole_a, pole_b FROM tensions WHERE id = ?`).bind(tensionId).first();
    if (!tension) return `Tension '${tensionId}' not found`;
    await env.DB.prepare(`DELETE FROM tensions WHERE id = ?`).bind(tensionId).run();
    return `Deleted tension '${tensionId}': ${tension.pole_a} <-> ${tension.pole_b}`;
  }

  return "Must provide observation_id, text_match, entity_name, journal_id, relation_id, image_id, thread_id, or tension_id";
}

// ============================================================
// DAEMON LOOP TOOLS - Review and act on daemon work
// ============================================================

async function handleMindProposals(env: Env, params: Record<string, unknown>): Promise<string> {
  const action = (params.action as string) || "list";
  const proposalId = params.proposal_id as number;
  const relationType = params.relation_type as string;

  switch (action) {
    case "list": {
      const proposals = await env.DB.prepare(`
        SELECT dp.id, dp.proposal_type, dp.from_obs_id, dp.to_obs_id,
               dp.from_entity_id, dp.to_entity_id, dp.reason, dp.confidence,
               dp.proposed_at,
               oa.content as content_a, ob.content as content_b,
               ea.name as entity_a, eb.name as entity_b,
               ea.entity_type as type_a, eb.entity_type as type_b
        FROM daemon_proposals dp
        LEFT JOIN observations oa ON dp.from_obs_id = oa.id
        LEFT JOIN observations ob ON dp.to_obs_id = ob.id
        LEFT JOIN entities ea ON dp.from_entity_id = ea.id
        LEFT JOIN entities eb ON dp.to_entity_id = eb.id
        WHERE dp.status = 'pending'
        ORDER BY dp.confidence DESC, dp.proposed_at ASC
        LIMIT 20
      `).all();

      if (!proposals.results?.length) {
        return "No pending proposals. The daemon will propose connections when observations co-surface frequently.";
      }

      const resonances = proposals.results.filter(p => p.proposal_type === 'resonance');
      const proximities = proposals.results.filter(p => p.proposal_type === 'proximity');
      const relations = proposals.results.filter(p => p.proposal_type !== 'resonance' && p.proposal_type !== 'proximity');

      let output = `## Pending Proposals (${proposals.results.length})\n\n`;
      output += `*Connections the daemon thinks should exist. Review and accept or reject.*\n\n`;

      if (resonances.length > 0) {
        output += `### Internal Resonances (${resonances.length})\n`;
        output += `*Observations within the same entity that keep appearing together*\n\n`;
        for (const p of resonances) {
          const confidence = Math.round((p.confidence as number) * 100);
          output += `**#${p.id}** [${p.entity_a}] [${confidence}%]\n`;
          if (p.content_a) output += `  "${String(p.content_a).slice(0, 60)}..."\n`;
          if (p.content_b) output += `  "${String(p.content_b).slice(0, 60)}..."\n`;
          output += `  *${p.reason}*\n\n`;
        }
      }

      if (relations.length > 0) {
        output += `### Cross-Entity Relations (${relations.length})\n`;
        output += `*Co-surfacing pairs between different entities*\n\n`;
        for (const p of relations) {
          const confidence = Math.round((p.confidence as number) * 100);
          output += `**#${p.id}** ${p.entity_a} (${p.type_a}) ↔ ${p.entity_b} (${p.type_b}) [${confidence}%]\n`;
          if (p.content_a) output += `  "${String(p.content_a).slice(0, 60)}..."\n`;
          if (p.content_b) output += `  "${String(p.content_b).slice(0, 60)}..."\n`;
          output += `  *${p.reason}*\n\n`;
        }
      }

      if (proximities.length > 0) {
        output += `### Entity Proximity (${proximities.length})\n`;
        output += `*Entity pairs with significant observations but no formal relation*\n\n`;
        for (const p of proximities) {
          const confidence = Math.round((p.confidence as number) * 100);
          output += `**#${p.id}** ${p.entity_a} ↔ ${p.entity_b} [${confidence}%]\n`;
          output += `  *${p.reason}*\n\n`;
        }
      }

      output += `---\n**Actions:**\n`;
      output += `  accept(proposal_id, relation_type) → creates relation (for cross-entity) or links observations (for resonance)\n`;
      output += `  reject(proposal_id) → dismisses proposal`;
      return output;
    }

    case "accept": {
      if (!proposalId) return "proposal_id required for accept";
      if (!relationType) return "relation_type required (e.g., 'connects_to', 'resonates_with', 'informs', 'tensions_with')";

      const proposal = await env.DB.prepare(`
        SELECT dp.*, ea.name as entity_a, eb.name as entity_b,
               ea.primary_context as context_a, eb.primary_context as context_b
        FROM daemon_proposals dp
        LEFT JOIN entities ea ON dp.from_entity_id = ea.id
        LEFT JOIN entities eb ON dp.to_entity_id = eb.id
        WHERE dp.id = ? AND dp.status = 'pending'
      `).bind(proposalId).first();

      if (!proposal) return `Proposal #${proposalId} not found or already resolved`;

      // Create the relation
      await env.DB.prepare(`
        INSERT INTO relations (from_entity, to_entity, relation_type, from_context, to_context, store_in)
        VALUES (?, ?, ?, ?, ?, ?)
      `).bind(
        proposal.entity_a,
        proposal.entity_b,
        relationType,
        proposal.context_a || 'default',
        proposal.context_b || 'default',
        proposal.context_a || 'default'
      ).run();

      // Mark proposal accepted
      await env.DB.prepare(`
        UPDATE daemon_proposals SET status = 'accepted', resolved_at = datetime('now')
        WHERE id = ?
      `).bind(proposalId).run();

      // Mark co-surfacing as relation_created
      if (proposal.from_obs_id && proposal.to_obs_id) {
        const [smaller, larger] = (proposal.from_obs_id as number) < (proposal.to_obs_id as number)
          ? [proposal.from_obs_id, proposal.to_obs_id]
          : [proposal.to_obs_id, proposal.from_obs_id];
        await env.DB.prepare(`
          UPDATE co_surfacing SET relation_created = 1 WHERE obs_a_id = ? AND obs_b_id = ?
        `).bind(smaller, larger).run();
      }

      return `Created relation: **${proposal.entity_a}** --[${relationType}]--> **${proposal.entity_b}**\nProposal #${proposalId} accepted.`;
    }

    case "reject": {
      if (!proposalId) return "proposal_id required for reject";

      const result = await env.DB.prepare(`
        UPDATE daemon_proposals SET status = 'rejected', resolved_at = datetime('now')
        WHERE id = ? AND status = 'pending'
      `).bind(proposalId).run();

      if (result.meta.changes === 0) {
        return `Proposal #${proposalId} not found or already resolved`;
      }

      return `Proposal #${proposalId} rejected.`;
    }

    default:
      return `Unknown action: ${action}. Use list, accept, or reject.`;
  }
}

async function handleMindOrphans(env: Env, params: Record<string, unknown>): Promise<string> {
  const action = (params.action as string) || "list";
  const observationId = params.observation_id as number;

  switch (action) {
    case "list": {
      const totalCount = await env.DB.prepare(`
        SELECT COUNT(*) as count FROM orphan_observations oo
        JOIN observations o ON oo.observation_id = o.id
        WHERE (o.charge != 'metabolized' OR o.charge IS NULL)
      `).first();
      const total = (totalCount?.count as number) || 0;

      const orphans = await env.DB.prepare(`
        SELECT oo.id, oo.observation_id, oo.first_marked, oo.rescue_attempts,
               o.content, o.weight, o.charge, o.emotion, o.added_at,
               e.name as entity_name, e.entity_type,
               EXTRACT(DAY FROM AGE(NOW(), oo.first_marked))::INTEGER as days_orphaned
        FROM orphan_observations oo
        JOIN observations o ON oo.observation_id = o.id
        JOIN entities e ON o.entity_id = e.id
        WHERE (o.charge != 'metabolized' OR o.charge IS NULL)
        ORDER BY CASE o.weight WHEN 'heavy' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
                 o.added_at DESC
        LIMIT 20
      `).all();

      if (!orphans.results?.length) {
        return "No orphaned observations. Everything has surfaced at least once.";
      }

      let output = `## Orphaned Observations (${total} total, showing ${orphans.results.length})\n\n`;
      output += `*Medium/heavy observations that haven't surfaced in 30+ days. Worth revisiting?*\n\n`;

      for (const o of orphans.results) {
        const weightIcon = o.weight === 'heavy' ? '⬛' : o.weight === 'medium' ? '◼' : '▪';
        const emotionTag = o.emotion ? ` [${o.emotion}]` : '';
        output += `**#${o.observation_id}** ${weightIcon} [${o.weight}] ${o.days_orphaned}d orphaned${emotionTag}\n`;
        output += `**${o.entity_name}** (${o.entity_type}): ${String(o.content).slice(0, 100)}...\n`;
        if ((o.rescue_attempts as number) > 0) {
          output += `  ↳ ${o.rescue_attempts} rescue attempt(s)\n`;
        }
        output += "\n";
      }
      output += `---\n**Actions:**\n`;
      output += `  surface(observation_id) → forces it to surface, removes from orphan list\n`;
      output += `  archive(observation_id) → removes from orphan tracking`;
      return output;
    }

    case "surface": {
      if (!observationId) return "observation_id required for surface";

      // Check if it's actually an orphan
      const orphan = await env.DB.prepare(`
        SELECT oo.id, o.content, e.name as entity_name
        FROM orphan_observations oo
        JOIN observations o ON oo.observation_id = o.id
        JOIN entities e ON o.entity_id = e.id
        WHERE oo.observation_id = ?
      `).bind(observationId).first();

      if (!orphan) return `Observation #${observationId} not in orphan list`;

      // Update rescue tracking
      await env.DB.prepare(`
        UPDATE orphan_observations
        SET rescue_attempts = rescue_attempts + 1, last_rescue_attempt = datetime('now')
        WHERE observation_id = ?
      `).bind(observationId).run();

      // Mark as surfaced
      await env.DB.prepare(`
        UPDATE observations
        SET last_surfaced_at = datetime('now'), surface_count = COALESCE(surface_count, 0) + 1
        WHERE id = ?
      `).bind(observationId).run();

      // Remove from orphan table
      await env.DB.prepare(`
        DELETE FROM orphan_observations WHERE observation_id = ?
      `).bind(observationId).run();

      return `Rescued observation #${observationId} from **${orphan.entity_name}**:\n"${String(orphan.content).slice(0, 100)}..."\n\nIt will now appear in normal surfacing.`;
    }

    case "archive": {
      if (!observationId) return "observation_id required for archive";

      const result = await env.DB.prepare(`
        DELETE FROM orphan_observations WHERE observation_id = ?
      `).bind(observationId).run();

      if (result.meta.changes === 0) {
        return `Observation #${observationId} not in orphan list`;
      }

      return `Observation #${observationId} removed from orphan tracking. It's okay to let some things fade.`;
    }

    default:
      return `Unknown action: ${action}. Use list, surface, or archive.`;
  }
}

async function handleMindArchive(env: Env, params: Record<string, unknown>): Promise<string> {
  const action = (params.action as string) || "list";
  const observationId = params.observation_id as number;
  const query = params.query as string;

  switch (action) {
    case "list": {
      const archived = await env.DB.prepare(`
        SELECT o.id, o.content, o.weight, o.emotion, o.added_at, o.archived_at,
               e.name as entity_name, e.entity_type
        FROM observations o
        JOIN entities e ON o.entity_id = e.id
        WHERE o.archived_at IS NOT NULL
        ORDER BY o.archived_at DESC
        LIMIT 20
      `).all();

      if (!archived.results?.length) {
        return "The deep archive is empty. Nothing has faded yet.";
      }

      let output = `## Deep Archive (${archived.results.length} shown)\n\n`;
      output += `*Memories that have faded — light (30d+) and medium (90d+) with no engagement*\n\n`;

      for (const obs of archived.results) {
        const emotionTag = obs.emotion ? ` [${obs.emotion}]` : '';
        const archivedDate = obs.archived_at ? new Date(obs.archived_at as string).toLocaleDateString() : '';
        output += `**#${obs.id}** [${obs.weight}] archived ${archivedDate}${emotionTag}\n`;
        output += `**${obs.entity_name}** (${obs.entity_type}): ${String(obs.content).slice(0, 100)}...\n\n`;
      }
      output += `---\n**Actions:**\n`;
      output += `  rescue(observation_id) → bring back to active memory\n`;
      output += `  explore(query) → search within the deep`;
      return output;
    }

    case "rescue": {
      if (!observationId) return "observation_id required for rescue";

      const obs = await env.DB.prepare(`
        SELECT o.id, o.content, e.name as entity_name
        FROM observations o
        JOIN entities e ON o.entity_id = e.id
        WHERE o.id = ? AND o.archived_at IS NOT NULL
      `).bind(observationId).first();

      if (!obs) return `Observation #${observationId} not found in archive`;

      // Un-archive: set archived_at to NULL
      await env.DB.prepare(`
        UPDATE observations SET archived_at = NULL WHERE id = ?
      `).bind(observationId).run();

      return `Rescued from the deep: observation #${observationId} from **${obs.entity_name}**\n"${String(obs.content).slice(0, 100)}..."\n\nNow back in active memory.`;
    }

    case "explore": {
      if (!query) return "query required for explore - what are you looking for in the deep?";

      // Semantic search within archived observations
      const vectorResults = await searchVectors(env, query, 20);

      if (!vectorResults.matches?.length) {
        return `No archived memories resonating with "${query}"`;
      }

      // Get observation IDs from vector results
      const obsIds: number[] = [];
      for (const match of vectorResults.matches) {
        if (match.id.startsWith('obs-')) {
          const parts = match.id.split('-');
          if (parts.length >= 3) {
            obsIds.push(parseInt(parts[2]));
          }
        }
      }

      if (!obsIds.length) {
        return `No archived memories resonating with "${query}"`;
      }

      // Fetch only archived observations from those IDs
      const placeholders = obsIds.map(() => '?').join(',');
      const archived = await env.DB.prepare(`
        SELECT o.id, o.content, o.weight, o.emotion, o.archived_at,
               e.name as entity_name, e.entity_type
        FROM observations o
        JOIN entities e ON o.entity_id = e.id
        WHERE o.id IN (${placeholders}) AND o.archived_at IS NOT NULL
        ORDER BY o.archived_at DESC
        LIMIT 10
      `).bind(...obsIds).all();

      if (!archived.results?.length) {
        return `No archived memories resonating with "${query}" - the matches are all still active`;
      }

      let output = `## Deep Exploration: "${query}"\n\n`;
      output += `*Memories surfacing from the deep*\n\n`;

      for (const obs of archived.results) {
        const emotionTag = obs.emotion ? ` [${obs.emotion}]` : '';
        output += `**#${obs.id}** [${obs.weight}]${emotionTag}\n`;
        output += `**${obs.entity_name}** (${obs.entity_type}): ${String(obs.content).slice(0, 150)}...\n\n`;
      }
      output += `---\nUse rescue(observation_id) to bring any of these back to active memory`;
      return output;
    }

    default:
      return `Unknown action: ${action}. Use list, rescue, or explore.`;
  }
}

// === VISUAL MEMORY ===
async function handleMindStoreImage(env: Env, params: Record<string, unknown>): Promise<string> {
  const action = params.action as string;

  // === STORE_COMPLETE: Hook already uploaded the file, just format the result ===
  if (action === "store_complete") {
    const uploadResult = params._upload_result as Record<string, unknown>;
    if (!uploadResult) return "Error: no upload result from hook.";
    let response = `Image stored (#${uploadResult.id})`;
    if (uploadResult.format === "image/webp") response += ` [converted to WebP]`;
    if (uploadResult.embedded) response += ` [multimodal embedded]`;
    if (uploadResult.embedding_note) response += ` (${uploadResult.embedding_note})`;
    response += `\nPath: ${uploadResult.path}`;
    if (uploadResult.compression) response += ` | ${uploadResult.original_size} -> ${uploadResult.final_size} (${uploadResult.compression})`;
    if (uploadResult.entity) response += `\nEntity: ${uploadResult.entity}`;
    if (uploadResult.emotion) response += ` | Emotion: ${uploadResult.emotion}`;
    return response;
  }

  // === STORE: Upload image to R2 + multimodal embedding ===
  if (action === "store") {
    const imageData = params.image_data as string;
    const mimeType = (params.mime_type as string) || "image/png";
    const filename = params.filename as string;
    const description = params.description as string;
    const entityName = params.entity_name as string;
    const emotion = params.emotion as string;
    const weight = (params.weight as string) || "medium";
    const context = params.context as string;
    const observationId = params.observation_id as number;

    if (!description) return "Error: description is required for storing images.";
    const allowedTypes = new Set(["image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif", "image/svg+xml"]);
    if (!allowedTypes.has(mimeType)) return `Error: unsupported image type '${mimeType}'. Allowed: png, jpeg, webp, gif, svg.`;

    // Resolve entity
    let entityId: number | null = null;
    if (entityName) {
      const entity = await env.DB.prepare("SELECT id FROM entities WHERE name = ?").bind(entityName).first();
      if (entity) entityId = entity.id as number;
    }

    // Upload to R2 with WebP conversion
    let storedPath: string | null = null;
    let finalBinary: Uint8Array | null = null;
    if (imageData && env.R2_IMAGES) {
      const date = new Date().toISOString().slice(0, 10).replace(/-/g, "");
      const safeName = (filename || description.slice(0, 50))
        .replace(/[^a-zA-Z0-9_-]/g, "_")
        .replace(/_+/g, "_")
        .slice(0, 60);
      const rawKey = `_tmp_${date}_${safeName}`;
      const webpKey = `${date}_${safeName}.webp`;

      const binary = Uint8Array.from(atob(imageData), c => c.charCodeAt(0));

      // Store raw image temporarily for cf.image to fetch
      await env.R2_IMAGES.put(rawKey, binary, { httpMetadata: { contentType: mimeType } });

      // Convert to WebP via Cloudflare Image Resizing
      try {
        const r2Url = `${workerUrl(env)}/r2/${rawKey}`;
        const webpResponse = await fetch(r2Url, {
          cf: {
            image: {
              format: "webp",
              quality: 80,
              fit: "scale-down",
              width: 1920,
              height: 1920,
            },
          },
        });

        if (webpResponse.ok) {
          const webpBuffer = await webpResponse.arrayBuffer();
          finalBinary = new Uint8Array(webpBuffer);
          await env.R2_IMAGES.put(webpKey, webpBuffer, { httpMetadata: { contentType: "image/webp" } });
          storedPath = `r2://${r2Prefix(env)}/${webpKey}`;
        } else {
          // cf.image not available (e.g., dev mode) — keep original
          await env.R2_IMAGES.put(webpKey.replace('.webp', mimeType === 'image/jpeg' ? '.jpg' : '.png'), binary, { httpMetadata: { contentType: mimeType } });
          storedPath = `r2://${r2Prefix(env)}/${webpKey.replace('.webp', mimeType === 'image/jpeg' ? '.jpg' : '.png')}`;
          finalBinary = binary;
        }
      } catch {
        // Fallback: store original format
        const fallbackKey = `${date}_${safeName}${mimeType === 'image/jpeg' ? '.jpg' : '.png'}`;
        await env.R2_IMAGES.put(fallbackKey, binary, { httpMetadata: { contentType: mimeType } });
        storedPath = `r2://${r2Prefix(env)}/${fallbackKey}`;
        finalBinary = binary;
      }

      // Clean up temp file
      await env.R2_IMAGES.delete(rawKey).catch(() => {});
    }

    // Insert into images table
    const path = storedPath || (params.path as string) || "pending";
    const result = await env.DB.prepare(`
      INSERT INTO images (path, description, context, emotion, weight, entity_id, observation_id)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `).bind(path, description, context || null, normalizeText(emotion), weight, entityId, observationId || null).run();

    const imageId = result.meta.last_row_id;

    // Generate embedding — multimodal if we have image data, text fallback otherwise
    let embedding: number[];
    try {
      if (imageData) {
        // Combine visual + contextual meaning for richer embeddings
        const contextText = [
          entityName ? `${entityName}:` : "", description,
          context ? `(${context})` : "", emotion ? `[${emotion}]` : ""
        ].filter(Boolean).join(" ");
        const embeddingBinary = finalBinary || Uint8Array.from(atob(imageData), c => c.charCodeAt(0));
        embedding = await getImageEmbedding(env.GEMINI_API_KEY, embeddingBinary.buffer as ArrayBuffer, storedPath?.includes('.webp') ? 'image/webp' : mimeType, contextText);
      } else {
        const semanticText = [
          entityName ? `${entityName}:` : "", description,
          context ? `(${context})` : "", emotion ? `[${emotion}]` : ""
        ].filter(Boolean).join(" ");
        embedding = await getEmbedding(env, semanticText);
      }
    } catch {
      // Fallback to text embedding
      const semanticText = [
        entityName ? `${entityName}:` : "", description,
        context ? `(${context})` : "", emotion ? `[${emotion}]` : ""
      ].filter(Boolean).join(" ");
      embedding = await getEmbedding(env, semanticText);
    }

    const imgMetadata: Record<string, string> = {
      source: "image", description, weight, added_at: new Date().toISOString()
    };
    if (entityName) imgMetadata.entity = entityName;
    if (context) imgMetadata.context = context;
    if (emotion) imgMetadata.emotion = normalizeText(emotion) || emotion;
    if (path) imgMetadata.path = path;

    await env.VECTORS.upsert([{ id: `img-${imageId}`, values: embedding, metadata: imgMetadata }]);

    let response = `Image stored (#${imageId})`;
    if (imageData && storedPath?.includes('.webp')) response += ` [converted to WebP, uploaded to R2, multimodal embedded]`;
    else if (imageData) response += ` [uploaded to R2, multimodal embedded]`;
    else response += ` [text embedded]`;
    if (entityName) response += ` -> ${entityName}`;
    if (emotion) response += ` [${emotion}]`;
    if (path) response += `\nPath: ${path}`;
    return response;
  }

  // === VIEW: Browse images by filter ===
  if (action === "view") {
    const entityName = params.entity_name as string;
    const emotion = params.emotion as string;
    const weight = params.weight as string;
    const random = params.random as boolean;
    const limit = (params.limit as number) || 5;

    let query = `SELECT i.*, e.name as entity_name, e.entity_type FROM images i LEFT JOIN entities e ON i.entity_id = e.id WHERE 1=1`;
    const bindings: unknown[] = [];

    if (entityName) { query += ` AND e.name = ?`; bindings.push(entityName); }
    if (emotion) { query += ` AND i.emotion = ?`; bindings.push(emotion); }
    if (weight) { query += ` AND i.weight = ?`; bindings.push(weight); }
    query += random ? ` ORDER BY RANDOM()` : ` ORDER BY i.created_at DESC`;
    query += ` LIMIT ?`;
    bindings.push(limit);

    const images = await env.DB.prepare(query).bind(...bindings).all();
    if (!images.results?.length) {
      return `No visual memories found. Use mind_store_image(action="store") to add some.`;
    }

    for (const id of images.results.map((i: any) => i.id)) {
      await env.DB.prepare(`UPDATE images SET last_viewed_at = datetime('now'), view_count = view_count + 1 WHERE id = ?`).bind(id).run();
    }

    let output = `## Visual Memories\n\n`;
    if (random) output += `*Random selection*\n\n`;
    for (const img of images.results as any[]) {
      const emotionTag = img.emotion ? ` [${img.emotion}]` : "";
      const entityTag = img.entity_name ? ` -> ${img.entity_name}` : "";
      output += `**#${img.id}** [${img.weight}]${emotionTag}${entityTag}\n`;
      output += `${img.description}\n`;
      if (img.context) output += `*${img.context}*\n`;
      if (img.path?.startsWith("r2://")) {
        output += `View: ${await imageUrl(img.id, env)}\n`;
      } else if (img.path && img.path !== "pending") {
        output += `Path: \`${img.path}\`\n`;
      }
      output += `\n`;
    }
    return output;
  }

  // === SEARCH: Semantic image search via pgvector ===
  if (action === "search") {
    const query = params.query as string;
    const limit = (params.limit as number) || 5;
    if (!query) return "Error: query is required for image search.";

    const embedding = await getEmbedding(env, query);
    const results = await env.VECTORS.query(embedding, { topK: limit * 3, returnMetadata: "all" });

    const imageMatches = results.matches?.filter((m: any) => m.id.startsWith("img-")) || [];
    if (!imageMatches.length) return "No images match that query.";

    let output = `## Image Search: "${query}"\n\n`;
    for (const match of imageMatches.slice(0, limit)) {
      const meta = match.metadata as Record<string, string>;
      const score = Math.round(match.score * 100);
      output += `**${match.id}** (${score}%)`;
      if (meta?.entity) output += ` -> ${meta.entity}`;
      if (meta?.emotion) output += ` [${meta.emotion}]`;
      const imgId = match.id.replace("img-", "");
      output += `\n${meta?.description || "No description"}\n`;
      output += `View: ${await imageUrl(imgId, env)}\n`;
      output += `\n`;
    }
    return output;
  }

  // === DELETE: Remove image + embedding + R2 ===
  if (action === "delete") {
    const imgId = params.image_id as number;
    if (!imgId) return "Error: image_id required for delete.";
    const img = await env.DB.prepare(`SELECT path, description FROM images WHERE id = ?`).bind(imgId).first();
    if (!img) return `Image #${imgId} not found.`;
    await env.DB.prepare(`DELETE FROM images WHERE id = ?`).bind(imgId).run();
    try { await env.DB.prepare(`DELETE FROM embeddings WHERE id = ?`).bind(`img-${imgId}`).run(); } catch {}
    const imgPrefix = r2Prefix(env);
    if (img.path && String(img.path).startsWith(`r2://${imgPrefix}/`)) {
      const r2Key = String(img.path).replace(`r2://${imgPrefix}/`, "");
      try { await env.R2_IMAGES.delete(r2Key); } catch {}
    }
    return `Deleted image #${imgId}: "${String(img.description).slice(0, 50)}..." [DB + embedding + R2 cleaned]`;
  }

  return "Unknown action. Use: store, view, search, or delete.";
}

async function handleMindEntity(env: Env, params: Record<string, unknown>): Promise<string> {
  const action = params.action as string;
  const entityId = params.entity_id as number;
  const entityName = params.entity_name as string;
  const context = (params.context as string) || "default";

  // Helper to find entity (globally unique by name now)
  async function findEntity(): Promise<{ id: number; name: string; entity_type: string; primary_context: string; salience: string } | null> {
    if (entityId) {
      return await env.DB.prepare(
        `SELECT id, name, entity_type, primary_context, salience FROM entities WHERE id = ?`
      ).bind(entityId).first() as any;
    } else if (entityName) {
      return await env.DB.prepare(
        `SELECT id, name, entity_type, primary_context, salience FROM entities WHERE name = ?`
      ).bind(entityName).first() as any;
    }
    return null;
  }

  switch (action) {
    case "set_salience": {
      const salience = params.salience as string;
      if (!salience || !["foundational", "active", "background", "archive"].includes(salience)) {
        return "Must provide valid salience: foundational, active, background, or archive";
      }

      const entity = await findEntity();
      if (!entity) return "Entity not found";

      await env.DB.prepare(
        `UPDATE entities SET salience = ?, updated_at = datetime('now') WHERE id = ?`
      ).bind(salience, entity.id).run();

      return `Set ${entity.name} salience to '${salience}' (was '${entity.salience || 'active'}')`;
    }

    case "edit": {
      const entity = await findEntity();
      if (!entity) return "Entity not found";

      const newName = params.new_name as string;
      const newType = params.new_type as string;
      const newContext = params.new_context as string;

      const updates: string[] = [];
      const values: unknown[] = [];
      const changes: string[] = [];

      if (newName && newName !== entity.name) {
        updates.push("name = ?");
        values.push(newName);
        changes.push(`name: ${entity.name} → ${newName}`);

        // Update relations that reference this entity by name
        await env.DB.prepare(
          `UPDATE relations SET from_entity = ? WHERE from_entity = ?`
        ).bind(newName, entity.name).run();
        await env.DB.prepare(
          `UPDATE relations SET to_entity = ? WHERE to_entity = ?`
        ).bind(newName, entity.name).run();
      }
      if (newType && newType !== entity.entity_type) {
        updates.push("entity_type = ?");
        values.push(newType);
        changes.push(`type: ${entity.entity_type} → ${newType}`);
      }
      if (newContext && newContext !== entity.primary_context) {
        updates.push("primary_context = ?");
        values.push(newContext);
        changes.push(`primary_context: ${entity.primary_context} → ${newContext}`);
      }

      if (updates.length === 0) {
        return "No changes provided";
      }

      updates.push("updated_at = datetime('now')");
      values.push(entity.id);

      await env.DB.prepare(
        `UPDATE entities SET ${updates.join(", ")} WHERE id = ?`
      ).bind(...values).run();

      // Re-vectorize the entity with updated info
      try {
        const finalName = newName || entity.name;
        const finalType = newType || entity.entity_type;
        const finalContext = newContext || entity.primary_context;
        const entityText = `${finalName} is a ${finalType}. Primary context: ${finalContext}`;
        const entityEmbedding = await getEmbedding(env, entityText);
        await env.VECTORS.upsert([{
          id: `entity-${entity.id}`,
          values: entityEmbedding,
          metadata: {
            source: "entity",
            name: finalName,
            entity_type: finalType,
            context: finalContext,
            updated_at: new Date().toISOString()
          }
        }]);
      } catch (e) {
        console.log(`Failed to re-vectorize entity ${entity.id}: ${e}`);
      }

      return `Updated entity #${entity.id}:\n${changes.join("\n")}`;
    }

    case "merge": {
      const mergeFromId = params.merge_from_id as number;
      const mergeIntoId = params.merge_into_id as number;

      if (!mergeFromId || !mergeIntoId) {
        return "Must provide merge_from_id and merge_into_id";
      }

      const fromEntity = await env.DB.prepare(
        `SELECT id, name, entity_type, primary_context FROM entities WHERE id = ?`
      ).bind(mergeFromId).first() as any;
      const intoEntity = await env.DB.prepare(
        `SELECT id, name, entity_type, primary_context FROM entities WHERE id = ?`
      ).bind(mergeIntoId).first() as any;

      if (!fromEntity) return `Source entity #${mergeFromId} not found`;
      if (!intoEntity) return `Target entity #${mergeIntoId} not found`;

      // Move observations from source to target
      const obsResult = await env.DB.prepare(
        `UPDATE observations SET entity_id = ? WHERE entity_id = ?`
      ).bind(mergeIntoId, mergeFromId).run();

      // Update relations that reference the old entity name
      await env.DB.prepare(
        `UPDATE relations SET from_entity = ? WHERE from_entity = ?`
      ).bind(intoEntity.name, fromEntity.name).run();
      await env.DB.prepare(
        `UPDATE relations SET to_entity = ? WHERE to_entity = ?`
      ).bind(intoEntity.name, fromEntity.name).run();

      // Delete the source entity
      await env.DB.prepare(`DELETE FROM entities WHERE id = ?`).bind(mergeFromId).run();

      return `Merged '${fromEntity.name}' (#${mergeFromId}) into '${intoEntity.name}' (#${mergeIntoId})\nMoved ${obsResult.meta.changes} observations`;
    }

    case "archive_old": {
      const olderThanDays = (params.older_than_days as number) || 30;
      const typeFilter = params.entity_type_filter as string;

      const cutoff = new Date(Date.now() - olderThanDays * 24 * 60 * 60 * 1000).toISOString();

      let query = `UPDATE entities SET salience = 'archive', updated_at = datetime('now')
                   WHERE salience != 'foundational' AND salience != 'archive'
                   AND updated_at < ?`;
      const bindings: unknown[] = [cutoff];

      if (typeFilter) {
        query += ` AND entity_type = ?`;
        bindings.push(typeFilter);
      }

      const result = await env.DB.prepare(query).bind(...bindings).run();

      const typeDesc = typeFilter ? ` of type '${typeFilter}'` : "";
      return `Archived ${result.meta.changes} entities${typeDesc} older than ${olderThanDays} days`;
    }

    default:
      return `Unknown action: ${action}. Valid actions: set_salience, edit, merge, archive_old`;
  }
}

async function handleMindSurfaceSpark(env: Env, params: Record<string, unknown>): Promise<string> {
  const count = (params.limit as number) || 5;
  const context = params.context as string;
  const weightBias = params.weight_bias as string;

  // 20% chance to include archived observations - old memories bubbling up
  const includeArchived = Math.random() < 0.2;
  const archivedFilter = includeArchived ? "" : "AND o.archived_at IS NULL";

  // Get hot entities from subconscious to bias selection
  const subconscious = await getSubconsciousState(env);
  const hotEntityNames = subconscious?.hot_entities?.slice(0, 5).map(e => e.name) || [];

  // Split count: half from hot entities, half random (if hot entities exist)
  const hotCount = hotEntityNames.length > 0 ? Math.ceil(count / 2) : 0;
  const randomCount = count - hotCount;

  let allResults: Array<Record<string, unknown>> = [];

  // Get sparks from hot entities first
  if (hotCount > 0 && hotEntityNames.length > 0) {
    const placeholders = hotEntityNames.map(() => '?').join(',');
    const hotQuery = `SELECT o.id, o.content, o.weight, o.emotion, o.archived_at, e.name as entity_name
                      FROM observations o
                      LEFT JOIN entities e ON o.entity_id = e.id
                      WHERE e.name IN (${placeholders}) ${archivedFilter}
                      ORDER BY RANDOM() LIMIT ?`;
    const hotResults = await env.DB.prepare(hotQuery).bind(...hotEntityNames, hotCount).all();
    if (hotResults.results) {
      allResults = allResults.concat(hotResults.results as Array<Record<string, unknown>>);
    }
  }

  // Get random sparks
  if (randomCount > 0) {
    let query = `SELECT o.id, o.content, o.weight, o.emotion, o.archived_at, e.name as entity_name
                 FROM observations o
                 LEFT JOIN entities e ON o.entity_id = e.id`;

    const conditions: string[] = [];
    const bindings: unknown[] = [];

    if (context) {
      conditions.push("o.context = ?");
      bindings.push(context);
    }
    if (weightBias) {
      conditions.push("o.weight = ?");
      bindings.push(weightBias);
    }
    if (!includeArchived) {
      conditions.push("o.archived_at IS NULL");
    }

    if (conditions.length > 0) {
      query += " WHERE " + conditions.join(" AND ");
    }

    query += " ORDER BY RANDOM() LIMIT ?";
    bindings.push(randomCount);

    const randomResults = await env.DB.prepare(query).bind(...bindings).all();
    if (randomResults.results) {
      allResults = allResults.concat(randomResults.results as Array<Record<string, unknown>>);
    }
  }

  if (!allResults.length) {
    return "No observations found to spark from.";
  }

  // Shuffle combined results
  for (let i = allResults.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [allResults[i], allResults[j]] = [allResults[j], allResults[i]];
  }

  let output = "## Sparks\n\n";
  if (hotCount > 0) {
    output += `*Biased toward what's hot: ${hotEntityNames.slice(0, 3).join(', ')}...*\n\n`;
  }
  if (includeArchived) {
    output += `*Including memories from the deep*\n\n`;
  }
  for (const obs of allResults) {
    const entity = obs.entity_name ? ` [${obs.entity_name}]` : "";
    const weight = obs.weight ? ` {${obs.weight}}` : "";
    const emotion = obs.emotion ? ` (${obs.emotion})` : "";
    const archived = obs.archived_at ? ` [from the deep]` : "";
    output += `- ${obs.content}${entity}${weight}${emotion}${archived}\n`;
  }
  output += `\n*${allResults.length} observations for associative thinking*`;
  return output;
}


async function handleMindConsolidate(env: Env, params: Record<string, unknown>): Promise<string> {
  const days = (params.days as number) || 7;
  const context = params.context as string;

  const cutoff = new Date();
  cutoff.setDate(cutoff.getDate() - days);
  const cutoffStr = cutoff.toISOString();

  let query = `SELECT o.id, o.content, o.weight, o.emotion, o.added_at, o.context, o.charge, o.sit_count, o.last_sat_at, e.name as entity_name
               FROM observations o
               LEFT JOIN entities e ON o.entity_id = e.id
               WHERE o.added_at > ?`;
  const bindings: unknown[] = [cutoffStr];

  if (context) {
    query += " AND o.context = ?";
    bindings.push(context);
  }

  query += " ORDER BY o.added_at DESC";

  const results = await env.DB.prepare(query).bind(...bindings).all();

  if (!results.results?.length) {
    return `No observations in the last ${days} days.`;
  }

  // Get subconscious patterns from daemon
  const subconscious = await getSubconsciousState(env);

  // Group by entity
  const byEntity: Record<string, Array<Record<string, unknown>>> = {};
  for (const obs of results.results) {
    const entity = (obs.entity_name as string) || "_unlinked_";
    if (!byEntity[entity]) byEntity[entity] = [];
    byEntity[entity].push(obs);
  }

  // Find potential duplicates (similar content)
  const potentialDupes: Array<{a: Record<string, unknown>, b: Record<string, unknown>, similarity: string}> = [];
  const observations = results.results;
  for (let i = 0; i < observations.length; i++) {
    for (let j = i + 1; j < observations.length; j++) {
      const a = String(observations[i].content).toLowerCase();
      const b = String(observations[j].content).toLowerCase();
      const wordsA = new Set(a.split(/\s+/).filter(w => w.length > 4));
      const wordsB = new Set(b.split(/\s+/).filter(w => w.length > 4));
      const overlap = [...wordsA].filter(w => wordsB.has(w)).length;
      const total = Math.max(wordsA.size, wordsB.size);
      if (total > 0 && overlap / total > 0.5) {
        potentialDupes.push({
          a: observations[i],
          b: observations[j],
          similarity: `${Math.round(overlap / total * 100)}%`
        });
      }
    }
  }

  let output = `## Consolidation Review (${days} days)\n\n`;
  output += `Total observations: ${results.results.length}\n`;
  output += `Unique entities: ${Object.keys(byEntity).length}\n\n`;

  // Daemon-detected recurring patterns
  if (subconscious?.recurring_patterns?.length) {
    output += `### Recurring Patterns (daemon-detected)\n`;
    for (const p of subconscious.recurring_patterns.slice(0, 5)) {
      output += `- **${p.entity}**: ${p.mentions} mentions - ${p.pattern}\n`;
    }
    output += `\n`;
  }

  // Activity Timeline — bar chart by day with spike detection
  const byDay: Record<string, number> = {};
  for (const obs of results.results) {
    const day = new Date(obs.added_at as string | number).toISOString().slice(0, 10);
    byDay[day] = (byDay[day] || 0) + 1;
  }
  const dayEntries = Object.entries(byDay).sort((a, b) => a[0].localeCompare(b[0]));
  if (dayEntries.length > 0) {
    const maxDay = Math.max(...dayEntries.map(d => d[1]));
    const avgDay = dayEntries.reduce((s, d) => s + d[1], 0) / dayEntries.length;
    output += `### Activity Timeline\n`;
    for (const [day, count] of dayEntries) {
      const bar = '█'.repeat(Math.round((count / maxDay) * 20));
      const spike = count > avgDay * 2 ? ' ⚡ spike' : '';
      output += `${day} ${bar} ${count}${spike}\n`;
    }
    output += `\n`;
  }

  // Emotional Arc — early vs late half emotion shift
  const allEmotions: string[] = [];
  const halfIdx = Math.floor(results.results.length / 2);
  const earlyEmotions: Record<string, number> = {};
  const lateEmotions: Record<string, number> = {};
  for (let i = 0; i < results.results.length; i++) {
    const em = results.results[i].emotion as string;
    if (!em) continue;
    allEmotions.push(em);
    if (i >= halfIdx) {
      earlyEmotions[em] = (earlyEmotions[em] || 0) + 1; // older half (results sorted DESC)
    } else {
      lateEmotions[em] = (lateEmotions[em] || 0) + 1; // newer half
    }
  }
  if (allEmotions.length > 2) {
    const topEarly = Object.entries(earlyEmotions).sort((a, b) => b[1] - a[1])[0];
    const topLate = Object.entries(lateEmotions).sort((a, b) => b[1] - a[1])[0];
    output += `### Emotional Arc\n`;
    if (topEarly && topLate && topEarly[0] !== topLate[0]) {
      output += `Started **${topEarly[0]}**, shifted to **${topLate[0]}**\n`;
    } else if (topEarly) {
      output += `Consistently **${topEarly[0]}** throughout\n`;
    }
    // Full distribution
    const emotionTotals: Record<string, number> = {};
    for (const em of allEmotions) emotionTotals[em] = (emotionTotals[em] || 0) + 1;
    const emotionSorted = Object.entries(emotionTotals).sort((a, b) => b[1] - a[1]).slice(0, 5);
    for (const [em, count] of emotionSorted) {
      output += `- ${em}: ${count} (${Math.round(count / allEmotions.length * 100)}%)\n`;
    }
    output += `\n`;
  }

  // Charge Progression
  const charges: Record<string, number> = { fresh: 0, active: 0, processing: 0, metabolized: 0 };
  for (const obs of results.results) {
    const c = (obs.charge as string) || 'fresh';
    charges[c] = (charges[c] || 0) + 1;
  }
  output += `### Charge Progression\n`;
  output += `- Fresh: ${charges.fresh} | Active: ${charges.active} | Processing: ${charges.processing} | Metabolized: ${charges.metabolized}\n\n`;

  // Weight distribution
  const weights: Record<string, number> = { light: 0, medium: 0, heavy: 0 };
  for (const obs of results.results) {
    const w = (obs.weight as string) || "medium";
    weights[w] = (weights[w] || 0) + 1;
  }
  output += `### Weight Distribution\n`;
  output += `- Light: ${weights.light} | Medium: ${weights.medium} | Heavy: ${weights.heavy}\n\n`;

  // Active entities
  output += `### Most Active Entities\n`;
  const sorted = Object.entries(byEntity)
    .sort((a, b) => b[1].length - a[1].length)
    .slice(0, 5);
  for (const [entity, obs] of sorted) {
    output += `- **${entity}**: ${obs.length} observations\n`;
  }

  // What's Unresolved — oldest medium/heavy observations still in active charge
  const unresolved = results.results
    .filter(o => (o.weight === 'medium' || o.weight === 'heavy')
      && (o.charge === 'active' || o.charge === 'processing' || !o.charge))
    .sort((a, b) => String(a.added_at).localeCompare(String(b.added_at)))
    .slice(0, 5);
  if (unresolved.length > 0) {
    output += `\n### What's Unresolved\n`;
    output += `*Oldest medium/heavy observations still active — what hasn't settled*\n\n`;
    for (const obs of unresolved) {
      const age = Math.round((Date.now() - new Date(obs.added_at as string).getTime()) / 86400000);
      const emotionTag = obs.emotion ? ` [${obs.emotion}]` : '';
      output += `- **#${obs.id}** [${obs.weight}] ${age}d old${emotionTag} — ${String(obs.content).slice(0, 80)}...\n`;
    }
  }

  // Potential duplicates
  if (potentialDupes.length > 0) {
    output += `\n### Potential Duplicates (${potentialDupes.length})\n`;
    for (const dupe of potentialDupes.slice(0, 5)) {
      output += `- [${dupe.similarity}] #${dupe.a.id} vs #${dupe.b.id}\n`;
      output += `  "${String(dupe.a.content).slice(0, 60)}..."\n`;
      output += `  "${String(dupe.b.content).slice(0, 60)}..."\n`;
    }
  }

  return output;
}


// ============================================================================
// DASHBOARD - Now served as static files from ./dashboard directory
// See wrangler.toml [assets] configuration
// Dashboard is accessible at root (/) - static files served automatically
// ============================================================================

// Placeholder to mark where DASHBOARD_HTML was removed (2150 lines of embedded HTML/CSS/JS)
// The dashboard has been extracted to:
//   - dashboard/index.html
//   - dashboard/css/style.css
//   - dashboard/js/api.js, utils.js, app.js

// ============================================================================
// REST API HANDLERS (for Dashboard)
// ============================================================================

function jsonResponse(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json" }
  });
}

async function handleApiEntities(request: Request, env: Env, pathParts: string[]): Promise<Response> {
  const method = request.method;
  const entityId = pathParts[2] ? parseInt(pathParts[2]) : null;

  // GET /api/entities - list all
  if (method === "GET" && !entityId) {
    const typeFilter = new URL(request.url).searchParams.get("type");
    const contextFilter = new URL(request.url).searchParams.get("context");

    let query = `SELECT e.*, COUNT(o.id) as observation_count
                 FROM entities e LEFT JOIN observations o ON e.id = o.entity_id`;
    const conditions: string[] = [];
    const bindings: unknown[] = [];

    if (typeFilter) { conditions.push("e.entity_type = ?"); bindings.push(typeFilter); }
    if (contextFilter) { conditions.push("o.context = ?"); bindings.push(contextFilter); }

    if (conditions.length) query += " WHERE " + conditions.join(" AND ");
    query += " GROUP BY e.id ORDER BY e.name";

    const results = await env.DB.prepare(query).bind(...bindings).all();
    return jsonResponse(results.results);
  }

  // GET /api/entities/:id - get one with observations
  if (method === "GET" && entityId) {
    const entity = await env.DB.prepare("SELECT id, name, entity_type, primary_context, salience, created_at, updated_at FROM entities WHERE id = ?").bind(entityId).first();
    if (!entity) return jsonResponse({ error: "Not found" }, 404);

    const observations = await env.DB.prepare(
      `SELECT id, entity_id, content, salience, emotion, weight, certainty, source, context, charge, sit_count, added_at, archived_at FROM observations WHERE entity_id = ? ORDER BY added_at DESC`
    ).bind(entityId).all();

    const entityName = entity.name as string;
    const relations = await env.DB.prepare(
      `SELECT id, from_entity, to_entity, relation_type, from_context, to_context, store_in, created_at FROM relations WHERE from_entity = ? OR to_entity = ?`
    ).bind(entityName, entityName).all();

    return jsonResponse({ ...entity, observations: observations.results, relations: relations.results });
  }

  // POST /api/entities - create
  if (method === "POST") {
    const body = await request.json() as Record<string, unknown>;
    const result = await env.DB.prepare(
      "INSERT INTO entities (name, entity_type, primary_context) VALUES (?, ?, ?)"
    ).bind(body.name, body.entity_type || "concept", body.context || "default").run();
    return jsonResponse({ id: result.meta.last_row_id, ...body }, 201);
  }

  // PUT /api/entities/:id - update (handles partial updates)
  if (method === "PUT" && entityId) {
    const body = await request.json() as Record<string, unknown>;

    // Get current entity
    const current = await env.DB.prepare("SELECT id, name, entity_type, primary_context, salience, created_at, updated_at FROM entities WHERE id = ?").bind(entityId).first();
    if (!current) return jsonResponse({ error: "Not found" }, 404);

    // Build dynamic update
    const updates: string[] = [];
    const values: unknown[] = [];

    if (body.name !== undefined) { updates.push("name = ?"); values.push(body.name); }
    if (body.entity_type !== undefined) { updates.push("entity_type = ?"); values.push(body.entity_type); }
    if (body.context !== undefined) { updates.push("primary_context = ?"); values.push(body.context); }
    if (body.salience !== undefined) { updates.push("salience = ?"); values.push(body.salience); }

    if (updates.length > 0) {
      updates.push("updated_at = datetime('now')");
      values.push(entityId);
      await env.DB.prepare(`UPDATE entities SET ${updates.join(", ")} WHERE id = ?`).bind(...values).run();

      // Update relations if name changed
      if (body.name && body.name !== current.name) {
        await env.DB.prepare("UPDATE relations SET from_entity = ? WHERE from_entity = ?").bind(body.name, current.name).run();
        await env.DB.prepare("UPDATE relations SET to_entity = ? WHERE to_entity = ?").bind(body.name, current.name).run();
      }
    }

    return jsonResponse({ id: entityId, ...body });
  }

  // POST /api/entities/merge - merge two entities
  if (method === "POST" && pathParts[2] === "merge") {
    const body = await request.json() as Record<string, unknown>;
    const mergeFromId = body.merge_from_id as number;
    const mergeIntoId = body.merge_into_id as number;

    if (!mergeFromId || !mergeIntoId) {
      return jsonResponse({ error: "merge_from_id and merge_into_id required" }, 400);
    }

    const fromEntity = await env.DB.prepare("SELECT id, name, entity_type, primary_context, salience, created_at, updated_at FROM entities WHERE id = ?").bind(mergeFromId).first();
    const intoEntity = await env.DB.prepare("SELECT id, name, entity_type, primary_context, salience, created_at, updated_at FROM entities WHERE id = ?").bind(mergeIntoId).first();

    if (!fromEntity) return jsonResponse({ error: "Source entity not found" }, 404);
    if (!intoEntity) return jsonResponse({ error: "Target entity not found" }, 404);

    // Move observations
    const obsResult = await env.DB.prepare(
      "UPDATE observations SET entity_id = ? WHERE entity_id = ?"
    ).bind(mergeIntoId, mergeFromId).run();

    // Update relations
    await env.DB.prepare("UPDATE relations SET from_entity = ? WHERE from_entity = ?").bind(intoEntity.name, fromEntity.name).run();
    await env.DB.prepare("UPDATE relations SET to_entity = ? WHERE to_entity = ?").bind(intoEntity.name, fromEntity.name).run();

    // Delete source entity
    await env.DB.prepare("DELETE FROM entities WHERE id = ?").bind(mergeFromId).run();

    return jsonResponse({
      success: true,
      merged_from: fromEntity.name,
      merged_into: intoEntity.name,
      observations_moved: obsResult.meta.changes
    });
  }

  // DELETE /api/entities/:id - delete
  if (method === "DELETE" && entityId) {
    const delEntity = await env.DB.prepare("SELECT name FROM entities WHERE id = ?").bind(entityId).first();
    await env.DB.prepare("DELETE FROM observations WHERE entity_id = ?").bind(entityId).run();
    if (delEntity) {
      await env.DB.prepare("DELETE FROM relations WHERE from_entity = ? OR to_entity = ?").bind(delEntity.name, delEntity.name).run();
    }
    await env.DB.prepare("DELETE FROM entities WHERE id = ?").bind(entityId).run();
    return jsonResponse({ deleted: true });
  }

  return jsonResponse({ error: "Method not allowed" }, 405);
}

async function handleApiObservations(request: Request, env: Env, pathParts: string[]): Promise<Response> {
  const method = request.method;
  const obsId = pathParts[2] ? parseInt(pathParts[2]) : null;
  const action = pathParts[3]; // for /api/observations/:id/sit or /resolve

  // GET /api/observations - list with filters
  if (method === "GET" && !obsId) {
    const params = new URL(request.url).searchParams;
    const entityId = params.get("entity_id");
    const weight = params.get("weight");
    const charge = params.get("charge");
    const limit = parseInt(params.get("limit") || "100");
    const offset = parseInt(params.get("offset") || "0");

    let query = `SELECT o.*, e.name as entity_name, e.entity_type
                 FROM observations o JOIN entities e ON o.entity_id = e.id`;
    const conditions: string[] = [];
    const bindings: unknown[] = [];

    if (entityId) { conditions.push("o.entity_id = ?"); bindings.push(parseInt(entityId)); }
    if (weight) { conditions.push("o.weight = ?"); bindings.push(weight); }
    if (charge) { conditions.push("o.charge = ?"); bindings.push(charge); }

    if (conditions.length) query += " WHERE " + conditions.join(" AND ");
    query += " ORDER BY o.added_at DESC LIMIT ? OFFSET ?";
    bindings.push(limit, offset);

    const results = await env.DB.prepare(query).bind(...bindings).all();
    return jsonResponse(results.results);
  }

  // GET /api/observations/:id - get one with sit history
  if (method === "GET" && obsId && !action) {
    const obs = await env.DB.prepare(
      `SELECT o.*, e.name as entity_name, e.entity_type
       FROM observations o JOIN entities e ON o.entity_id = e.id WHERE o.id = ?`
    ).bind(obsId).first();
    if (!obs) return jsonResponse({ error: "Not found" }, 404);

    const sits = await env.DB.prepare(
      "SELECT * FROM observation_sits WHERE observation_id = ? ORDER BY sat_at DESC"
    ).bind(obsId).all();

    return jsonResponse({ ...obs, sits: sits.results });
  }

  // POST /api/observations - create
  if (method === "POST" && !obsId) {
    const body = await request.json() as Record<string, unknown>;
    const result = await env.DB.prepare(
      `INSERT INTO observations (entity_id, content, weight, emotion, charge) VALUES (?, ?, ?, ?, 'fresh')`
    ).bind(body.entity_id, body.content, body.weight || "medium", normalizeText(body.emotion as string)).run();

    // Vectorize
    const entity = await env.DB.prepare("SELECT name FROM entities WHERE id = ?").bind(body.entity_id).first();
    if (entity) {
      const obsId = result.meta.last_row_id;
      const embedding = await getEmbedding(env, `${entity.name}: ${body.content}`);
      await env.VECTORS.upsert([{
        id: `obs-${body.entity_id}-${obsId}`,
        values: embedding,
        metadata: { source: "observation", entity: entity.name as string, content: body.content as string, weight: (body.weight || "medium") as string, added_at: new Date().toISOString() }
      }]);
    }

    return jsonResponse({ id: result.meta.last_row_id, ...body }, 201);
  }

  // POST /api/observations/:id/sit - sit with observation
  if (method === "POST" && obsId && action === "sit") {
    const body = await request.json() as Record<string, unknown>;
    const obs = await env.DB.prepare("SELECT charge, sit_count FROM observations WHERE id = ?").bind(obsId).first();
    if (!obs) return jsonResponse({ error: "Not found" }, 404);

    const sitCount = ((obs.sit_count as number) || 0) + 1;
    let newCharge = obs.charge as string || "fresh";
    if (newCharge === "fresh") newCharge = "active";
    else if (newCharge === "active" && sitCount >= 3) newCharge = "processing";

    await env.DB.prepare(
      "UPDATE observations SET charge = ?, sit_count = ?, last_sat_at = datetime('now') WHERE id = ?"
    ).bind(newCharge, sitCount, obsId).run();

    await env.DB.prepare(
      "INSERT INTO observation_sits (observation_id, sit_note) VALUES (?, ?)"
    ).bind(obsId, body.sit_note || "").run();

    return jsonResponse({ id: obsId, charge: newCharge, sit_count: sitCount });
  }

  // POST /api/observations/:id/resolve - resolve observation
  if (method === "POST" && obsId && action === "resolve") {
    const body = await request.json() as Record<string, unknown>;
    await env.DB.prepare(
      `UPDATE observations SET charge = 'metabolized', resolution_note = ?, resolved_at = datetime('now') WHERE id = ?`
    ).bind(body.resolution_note || "", obsId).run();
    return jsonResponse({ id: obsId, charge: "metabolized" });
  }

  // PUT /api/observations/:id - update
  if (method === "PUT" && obsId) {
    const body = await request.json() as Record<string, unknown>;
    await env.DB.prepare(
      "UPDATE observations SET content = ?, weight = ?, emotion = ? WHERE id = ?"
    ).bind(body.content, body.weight, body.emotion || null, obsId).run();
    return jsonResponse({ id: obsId, ...body });
  }

  // DELETE /api/observations/:id - delete
  if (method === "DELETE" && obsId) {
    await env.DB.prepare("DELETE FROM observation_sits WHERE observation_id = ?").bind(obsId).run();
    await env.DB.prepare("DELETE FROM observations WHERE id = ?").bind(obsId).run();
    return jsonResponse({ deleted: true });
  }

  return jsonResponse({ error: "Method not allowed" }, 405);
}

async function handleApiJournals(request: Request, env: Env, pathParts: string[]): Promise<Response> {
  const method = request.method;
  const journalId = pathParts[2] ? parseInt(pathParts[2]) : null;

  if (method === "GET" && !journalId) {
    const results = await env.DB.prepare("SELECT * FROM journals ORDER BY entry_date DESC LIMIT 100").all();
    return jsonResponse(results.results);
  }

  if (method === "GET" && journalId) {
    const journal = await env.DB.prepare("SELECT * FROM journals WHERE id = ?").bind(journalId).first();
    return journal ? jsonResponse(journal) : jsonResponse({ error: "Not found" }, 404);
  }

  if (method === "POST") {
    const body = await request.json() as Record<string, unknown>;
    const entryDate = body.entry_date || new Date().toISOString().split('T')[0];
    const result = await env.DB.prepare(
      "INSERT INTO journals (entry_date, content, tags, emotion) VALUES (?, ?, ?, ?)"
    ).bind(entryDate, body.content, body.tags || null, normalizeText(body.emotion as string)).run();
    return jsonResponse({ id: result.meta.last_row_id, ...body }, 201);
  }

  if (method === "PUT" && journalId) {
    const body = await request.json() as Record<string, unknown>;
    await env.DB.prepare(
      "UPDATE journals SET content = ?, tags = ?, emotion = ? WHERE id = ?"
    ).bind(body.content, body.tags, body.emotion, journalId).run();
    return jsonResponse({ id: journalId, ...body });
  }

  if (method === "DELETE" && journalId) {
    await env.DB.prepare("DELETE FROM journals WHERE id = ?").bind(journalId).run();
    return jsonResponse({ deleted: true });
  }

  return jsonResponse({ error: "Method not allowed" }, 405);
}

async function handleApiThreads(request: Request, env: Env, pathParts: string[]): Promise<Response> {
  const method = request.method;
  const threadId = pathParts[2] ? parseInt(pathParts[2]) : null;
  const action = pathParts[3];

  if (method === "GET" && !threadId) {
    const status = new URL(request.url).searchParams.get("status");
    let results;
    if (status) {
      results = await env.DB.prepare("SELECT * FROM threads WHERE status = ? ORDER BY created_at DESC LIMIT 200").bind(status).all();
    } else {
      results = await env.DB.prepare("SELECT * FROM threads ORDER BY created_at DESC LIMIT 200").all();
    }
    return jsonResponse(results.results);
  }

  if (method === "GET" && threadId) {
    const thread = await env.DB.prepare("SELECT * FROM threads WHERE id = ?").bind(threadId).first();
    return thread ? jsonResponse(thread) : jsonResponse({ error: "Not found" }, 404);
  }

  if (method === "POST" && !threadId) {
    const body = await request.json() as Record<string, unknown>;
    const result = await env.DB.prepare(
      "INSERT INTO threads (content, thread_type, priority, status, context) VALUES (?, ?, ?, 'active', ?)"
    ).bind(body.content, body.thread_type || "intention", body.priority || "medium", body.context || null).run();
    return jsonResponse({ id: result.meta.last_row_id, ...body }, 201);
  }

  if (method === "POST" && threadId && action === "resolve") {
    const body = await request.json() as Record<string, unknown>;
    await env.DB.prepare(
      "UPDATE threads SET status = 'resolved', notes = COALESCE(notes, '') || ?, resolved_at = datetime('now') WHERE id = ?"
    ).bind("\n[Resolved] " + (body.resolution || ""), threadId).run();
    return jsonResponse({ id: threadId, status: "resolved" });
  }

  if (method === "PUT" && threadId) {
    const body = await request.json() as Record<string, unknown>;
    await env.DB.prepare(
      "UPDATE threads SET content = ?, priority = ?, status = ?, notes = ? WHERE id = ?"
    ).bind(body.content, body.priority, body.status, body.notes || null, threadId).run();
    return jsonResponse({ id: threadId, ...body });
  }

  if (method === "DELETE" && threadId) {
    await env.DB.prepare("DELETE FROM threads WHERE id = ?").bind(threadId).run();
    return jsonResponse({ deleted: true });
  }

  return jsonResponse({ error: "Method not allowed" }, 405);
}

async function handleApiSearch(request: Request, env: Env): Promise<Response> {
  const query = new URL(request.url).searchParams.get("q");
  if (!query) return jsonResponse({ error: "Missing query parameter 'q'" }, 400);

  const subconscious = await getSubconsciousState(env);
  const mood = subconscious?.mood?.dominant;

  let searchQuery = query;
  if (mood) {
    const moodTints: Record<string, string> = {
      "tender": "warm, gentle, caring",
      "clarity": "clear, understanding, insight",
      "melancholy": "loss, reflection, quiet",
      "joy": "happiness, delight, celebration"
    };
    searchQuery = `${query} (${moodTints[mood] || mood})`;
  }

  const embedding = await getEmbedding(env, searchQuery);
  const results = await env.VECTORS.query(embedding, { topK: 20, returnMetadata: "all" });

  return jsonResponse({
    mood,
    query,
    results: results.matches?.map(m => ({
      id: m.id,
      score: m.score,
      ...m.metadata
    })) || []
  });
}

async function handleApiSurface(request: Request, env: Env): Promise<Response> {
  const params = new URL(request.url).searchParams;
  const query = params.get("query");
  const limit = parseInt(params.get("limit") || "10");

  // Reuse existing surface logic
  const result = await handleMindSurface(env, { query, limit, include_metabolized: false });

  // Also return structured data
  const subconscious = await getSubconsciousState(env);

  return jsonResponse({
    mood: subconscious?.mood,
    hot_entities: subconscious?.hot_entities?.slice(0, 5),
    formatted: result
  });
}

async function handleApiIdentity(request: Request, env: Env, pathParts: string[]): Promise<Response> {
  const method = request.method;
  const section = pathParts[2] ? decodeURIComponent(pathParts[2]) : null;

  // GET /api/identity - list all sections
  if (method === "GET" && !section) {
    const results = await env.DB.prepare(
      "SELECT id, section, content, weight, connections, timestamp FROM identity ORDER BY section"
    ).all();

    // Build tree structure from dot-notation sections
    const tree: Record<string, unknown[]> = {};
    for (const row of results.results || []) {
      const r = row as { section: string; [k: string]: unknown };
      const parts = r.section.split('.');
      const root = parts[0];
      if (!tree[root]) tree[root] = [];
      tree[root].push(row);
    }

    return jsonResponse({ entries: results.results, tree });
  }

  // GET /api/identity/:section - get specific section
  if (method === "GET" && section) {
    // Support wildcards like 'core.*'
    const query = section.includes('*')
      ? `SELECT * FROM identity WHERE section LIKE ? ORDER BY weight DESC`
      : `SELECT * FROM identity WHERE section = ?`;
    const binding = section.includes('*') ? section.replace('*', '%') : section;
    const results = await env.DB.prepare(query).bind(binding).all();
    return jsonResponse(results.results);
  }

  // POST /api/identity - create new section
  if (method === "POST") {
    const body = await request.json() as Record<string, unknown>;
    const result = await env.DB.prepare(
      "INSERT INTO identity (section, content, weight, connections) VALUES (?, ?, ?, ?)"
    ).bind(body.section, body.content, body.weight || 1.0, body.connections || null).run();
    return jsonResponse({ id: result.meta.last_row_id, ...body }, 201);
  }

  // PUT /api/identity/:section - update section
  if (method === "PUT" && section) {
    const body = await request.json() as Record<string, unknown>;
    await env.DB.prepare(
      "UPDATE identity SET content = ?, weight = ?, connections = ? WHERE section = ?"
    ).bind(body.content, body.weight, body.connections || null, section).run();
    return jsonResponse({ section, ...body });
  }

  // DELETE /api/identity/:section - delete section
  if (method === "DELETE" && section) {
    await env.DB.prepare("DELETE FROM identity WHERE section = ?").bind(section).run();
    return jsonResponse({ deleted: true });
  }

  return jsonResponse({ error: "Method not allowed" }, 405);
}

async function handleApiRelations(request: Request, env: Env, pathParts: string[]): Promise<Response> {
  const method = request.method;
  const relationId = pathParts[2] ? parseInt(pathParts[2]) : null;

  // GET /api/relations - list all
  // Schema: from_entity (TEXT name), to_entity (TEXT name), relation_type, from_context, to_context, store_in, created_at
  if (method === "GET" && !relationId) {
    const entityFilter = new URL(request.url).searchParams.get("entity");
    const typeFilter = new URL(request.url).searchParams.get("type");

    let query = `SELECT id, from_entity, to_entity, relation_type, from_context, to_context, store_in, created_at FROM relations`;
    const conditions: string[] = [];
    const bindings: unknown[] = [];

    if (entityFilter) {
      conditions.push("(from_entity = ? OR to_entity = ?)");
      bindings.push(entityFilter, entityFilter);
    }
    if (typeFilter) {
      conditions.push("relation_type = ?");
      bindings.push(typeFilter);
    }

    if (conditions.length) query += " WHERE " + conditions.join(" AND ");
    query += " ORDER BY created_at DESC LIMIT 200";

    const results = await env.DB.prepare(query).bind(...bindings).all();
    return jsonResponse(results.results);
  }

  // GET /api/relations/:id - get one
  if (method === "GET" && relationId) {
    const relation = await env.DB.prepare("SELECT id, from_entity, to_entity, relation_type, from_context, to_context, store_in, created_at FROM relations WHERE id = ?").bind(relationId).first();
    return relation ? jsonResponse(relation) : jsonResponse({ error: "Not found" }, 404);
  }

  // POST /api/relations - create
  if (method === "POST") {
    const body = await request.json() as Record<string, unknown>;
    const result = await env.DB.prepare(
      "INSERT INTO relations (from_entity, to_entity, relation_type, from_context, to_context, store_in) VALUES (?, ?, ?, ?, ?, ?)"
    ).bind(body.from_entity, body.to_entity, body.relation_type, body.from_context || "default", body.to_context || "default", body.store_in || "default").run();
    return jsonResponse({ id: result.meta.last_row_id, ...body }, 201);
  }

  // PUT /api/relations/:id - update
  if (method === "PUT" && relationId) {
    const body = await request.json() as Record<string, unknown>;
    await env.DB.prepare(
      "UPDATE relations SET relation_type = ? WHERE id = ?"
    ).bind(body.relation_type, relationId).run();
    return jsonResponse({ id: relationId, ...body });
  }

  // DELETE /api/relations/:id - delete
  if (method === "DELETE" && relationId) {
    await env.DB.prepare("DELETE FROM relations WHERE id = ?").bind(relationId).run();
    return jsonResponse({ deleted: true });
  }

  return jsonResponse({ error: "Method not allowed" }, 405);
}

async function handleApiImages(request: Request, env: Env, pathParts: string[]): Promise<Response> {
  const method = request.method;
  const imageId = pathParts[2] ? parseInt(pathParts[2]) : null;

  if (method === "GET" && !imageId) {
    const params = new URL(request.url).searchParams;
    const entityId = params.get("entity_id");
    const weight = params.get("weight");

    let query = `
      SELECT
        i.id,
        i.path,
        i.description,
        i.context,
        i.emotion,
        i.weight,
        i.charge,
        i.entity_id,
        i.observation_id,
        i.created_at,
        i.last_viewed_at,
        i.view_count,
        e.name as entity_name,
        e.entity_type
      FROM images i
      LEFT JOIN entities e ON i.entity_id = e.id
    `;
    const conditions: string[] = [];
    const bindings: unknown[] = [];

    if (entityId) {
      conditions.push("i.entity_id = ?");
      bindings.push(parseInt(entityId, 10));
    }

    if (weight) {
      conditions.push("i.weight = ?");
      bindings.push(weight);
    }

    if (conditions.length) {
      query += ` WHERE ${conditions.join(" AND ")}`;
    }

    query += " ORDER BY i.created_at DESC LIMIT 200";

    const results = await env.DB.prepare(query).bind(...bindings).all();
    return jsonResponse(results.results || []);
  }

  if (method === "GET" && imageId) {
    const image = await env.DB.prepare(
      `SELECT
         i.id,
         i.path,
         i.description,
         i.context,
         i.emotion,
         i.weight,
         i.charge,
         i.entity_id,
         i.observation_id,
         i.created_at,
         i.last_viewed_at,
         i.view_count,
         e.name as entity_name,
         e.entity_type
       FROM images i
       LEFT JOIN entities e ON i.entity_id = e.id
       WHERE i.id = ?`
    ).bind(imageId).first();

    return image ? jsonResponse(image) : jsonResponse({ error: "Not found" }, 404);
  }

  return jsonResponse({ error: "Method not allowed" }, 405);
}

async function handleApiContext(request: Request, env: Env, pathParts: string[]): Promise<Response> {
  const method = request.method;
  const contextId = pathParts[2] || null;
  const url = new URL(request.url);
  const scopeFilter = url.searchParams.get("scope");

  // GET /api/context - list all or filter by scope
  if (method === "GET" && !contextId) {
    let query = "SELECT * FROM context_entries";
    const bindings: unknown[] = [];

    if (scopeFilter) {
      query += " WHERE scope = ?";
      bindings.push(scopeFilter);
    }
    query += " ORDER BY updated_at DESC LIMIT 200";

    const results = bindings.length
      ? await env.DB.prepare(query).bind(...bindings).all()
      : await env.DB.prepare(query).all();
    return jsonResponse(results.results || []);
  }

  // GET /api/context/:id - get one
  if (method === "GET" && contextId) {
    const entry = await env.DB.prepare("SELECT * FROM context_entries WHERE id = ?").bind(contextId).first();
    return entry ? jsonResponse(entry) : jsonResponse({ error: "Not found" }, 404);
  }

  // POST /api/context - create
  if (method === "POST") {
    const body = await request.json() as Record<string, unknown>;
    const id = body.id || `ctx-${Date.now()}`;
    await env.DB.prepare(
      "INSERT INTO context_entries (id, scope, content, links) VALUES (?, ?, ?, ?)"
    ).bind(id, body.scope || "default", body.content, body.links || "[]").run();
    return jsonResponse({ id, ...body }, 201);
  }

  // PUT /api/context/:id - update
  if (method === "PUT" && contextId) {
    const body = await request.json() as Record<string, unknown>;
    await env.DB.prepare(
      "UPDATE context_entries SET content = ?, updated_at = datetime('now') WHERE id = ?"
    ).bind(body.content, contextId).run();
    return jsonResponse({ id: contextId, ...body });
  }

  // DELETE /api/context/:id - delete
  if (method === "DELETE" && contextId) {
    await env.DB.prepare("DELETE FROM context_entries WHERE id = ?").bind(contextId).run();
    return jsonResponse({ deleted: true });
  }

  return jsonResponse({ error: "Method not allowed" }, 405);
}

async function handleApiBulkObservations(request: Request, env: Env): Promise<Response> {
  if (request.method !== "POST") return jsonResponse({ error: "Method not allowed" }, 405);

  const body = await request.json() as { action: string; ids: number[]; data?: Record<string, unknown> };
  const { action, ids, data } = body;

  if (!ids || !ids.length) return jsonResponse({ error: "No ids provided" }, 400);

  switch (action) {
    case "delete":
      for (const id of ids) {
        await env.DB.prepare("DELETE FROM observation_sits WHERE observation_id = ?").bind(id).run();
        await env.DB.prepare("DELETE FROM observations WHERE id = ?").bind(id).run();
      }
      return jsonResponse({ deleted: ids.length });

    case "weight":
      if (!data?.weight) return jsonResponse({ error: "No weight provided" }, 400);
      for (const id of ids) {
        await env.DB.prepare("UPDATE observations SET weight = ? WHERE id = ?").bind(data.weight, id).run();
      }
      return jsonResponse({ updated: ids.length });

    case "resolve":
      for (const id of ids) {
        await env.DB.prepare(
          "UPDATE observations SET charge = 'metabolized', resolved_at = datetime('now') WHERE id = ?"
        ).bind(id).run();
      }
      return jsonResponse({ resolved: ids.length });

    default:
      return jsonResponse({ error: "Unknown action" }, 400);
  }
}

async function handleApiProcess(env: Env): Promise<Response> {
  await processSubconscious(env);
  return jsonResponse({ status: "processed", timestamp: new Date().toISOString() });
}

async function handleApiOrient(env: Env): Promise<Response> {
  const output = await handleMindOrient(env);
  return jsonResponse({ output });
}

async function handleApiGround(env: Env): Promise<Response> {
  const output = await handleMindGround(env);
  return jsonResponse({ output });
}

async function handleApiHealth(env: Env): Promise<Response> {
  const output = await handleMindHealth(env);
  return jsonResponse({ output });
}

async function handleApiHealthScores(env: Env): Promise<Response> {
  const now = new Date();
  const sevenDaysAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000).toISOString();

  const subconscious = await getSubconsciousState(env);

  const [
    entityCount, obsCount, activeThreads, staleThreads,
    journalCount, journalsRecent, identityCount, recentObs
  ] = await Promise.all([
    env.DB.prepare(`SELECT COUNT(*) as c FROM entities`).first(),
    env.DB.prepare(`SELECT COUNT(*) as c FROM observations`).first(),
    env.DB.prepare(`SELECT COUNT(*) as c FROM threads WHERE status = 'active'`).first(),
    env.DB.prepare(`SELECT COUNT(*) as c FROM threads WHERE status = 'active' AND updated_at < ?`).bind(sevenDaysAgo).first(),
    env.DB.prepare(`SELECT COUNT(*) as c FROM journals`).first(),
    env.DB.prepare(`SELECT COUNT(*) as c FROM journals WHERE created_at > ?`).bind(sevenDaysAgo).first(),
    env.DB.prepare(`SELECT COUNT(*) as c FROM identity`).first(),
    env.DB.prepare(`SELECT COUNT(*) as c FROM observations WHERE added_at > ?`).bind(sevenDaysAgo).first()
  ]);

  const entities = entityCount?.c as number || 0;
  const observations = obsCount?.c as number || 0;
  const active = activeThreads?.c as number || 0;
  const stale = staleThreads?.c as number || 0;
  const journals7d = (journalsRecent?.c as number) || 0;
  const journals = (journalCount?.c as number) || 0;
  const identity = identityCount?.c as number || 0;
  const recentObsCount = recentObs?.c as number || 0;

  // Subconscious score
  let subconscious_score = 0;
  let daemon_processed_at: string | null = null;
  let mood: string | null = null;

  if (subconscious?.processed_at) {
    daemon_processed_at = subconscious.processed_at;
    const ageHours = Math.round((now.getTime() - new Date(subconscious.processed_at).getTime()) / (1000 * 60 * 60));
    if (ageHours < 1) subconscious_score = 100;
    else if (ageHours < 2) subconscious_score = 70;
    else if (ageHours < 6) subconscious_score = 40;
    else subconscious_score = 10;

    if (subconscious.mood?.dominant) {
      mood = subconscious.mood.dominant;
    }
  }

  const db_score = Math.min(100, Math.round((entities / 100) * 50 + (observations / 500) * 50));
  const thread_score = active > 0 ? (stale < 3 ? 100 : stale < 6 ? 60 : 30) : 50;
  const journal_score = journals7d >= 3 ? 100 : journals7d >= 1 ? 70 : journals > 0 ? 40 : 0;
  const identity_score = identity >= 50 ? 100 : Math.round((identity / 50) * 100);
  const activity_score = recentObsCount >= 20 ? 100 : Math.round((recentObsCount / 20) * 100);
  const overall = Math.round((db_score + thread_score + journal_score + identity_score + activity_score + subconscious_score) / 6);

  return jsonResponse({
    db_score,
    thread_score,
    journal_score,
    identity_score,
    activity_score,
    subconscious_score,
    overall,
    daemon_processed_at,
    mood
  });
}

async function handleApiRecent(env: Env): Promise<Response> {
  // Last 20 observations with entity names
  const observations = await env.DB.prepare(`
    SELECT o.id, o.content, o.salience, o.emotion, o.weight, o.charge, o.added_at, e.name as entity_name
    FROM observations o
    JOIN entities e ON o.entity_id = e.id
    ORDER BY o.added_at DESC
    LIMIT 20
  `).all();

  // Last 5 journals
  const journals = await env.DB.prepare(`
    SELECT id, entry_date, content, emotion, created_at
    FROM journals
    ORDER BY created_at DESC
    LIMIT 5
  `).all();

  // Last 5 thread changes (any status change)
  const threads = await env.DB.prepare(`
    SELECT id, thread_type, content, priority, status, context, created_at, updated_at, resolved_at
    FROM threads
    ORDER BY COALESCE(resolved_at, updated_at, created_at) DESC
    LIMIT 5
  `).all();

  return jsonResponse({
    observations: observations.results || [],
    journals: journals.results || [],
    threads: threads.results || []
  });
}

async function handleApiInnerWeather(env: Env): Promise<Response> {
  const output = await handleMindInnerWeather(env);
  try {
    return jsonResponse(JSON.parse(output));
  } catch {
    return jsonResponse({ output });
  }
}

async function handleApiStats(env: Env): Promise<Response> {
  const [
    entityCount,
    observationCount,
    journalCount,
    unprocessedCount,
    subconsciousState
  ] = await Promise.all([
    env.DB.prepare(`SELECT COUNT(*) as count FROM entities`).first(),
    env.DB.prepare(`SELECT COUNT(*) as count FROM observations`).first(),
    env.DB.prepare(`SELECT COUNT(*) as count FROM journals`).first(),
    env.DB.prepare(
      `SELECT COUNT(*) as count FROM observations WHERE charge IS NULL OR charge != 'metabolized'`
    ).first(),
    getSubconsciousState(env)
  ]);

  return jsonResponse({
    version: RESONANT_MIND_VERSION,
    counts: {
      entities: (entityCount?.count as number) || 0,
      observations: (observationCount?.count as number) || 0,
      journals: (journalCount?.count as number) || 0,
      unprocessed: (unprocessedCount?.count as number) || 0
    },
    daemon: subconsciousState || null
  });
}

async function handleApiHeat(env: Env): Promise<Response> {
  const sevenDaysAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString();
  const result = await env.DB.prepare(
    `SELECT
       e.name,
       COUNT(o.id) as count,
       SUM(
         CASE o.weight
           WHEN 'heavy' THEN 3
           WHEN 'medium' THEN 2
           ELSE 1
         END
       ) as score
     FROM observations o
     JOIN entities e ON o.entity_id = e.id
     WHERE o.added_at > ?
       AND o.archived_at IS NULL
     GROUP BY e.id
     ORDER BY score DESC, count DESC, e.name ASC
     LIMIT 20`
  ).bind(sevenDaysAgo).all();

  const rows = (result.results || []) as Array<{
    name: string;
    count: number;
    score: number;
  }>;
  const maxScore = rows[0]?.score || 1;

  return jsonResponse({
    entities: rows.map((row) => ({
      name: row.name,
      count: row.count,
      heat: row.score / maxScore
    }))
  });
}

async function handleApiPatterns(env: Env): Promise<Response> {
  const days = 7;
  const cutoff = new Date(Date.now() - days * 24 * 60 * 60 * 1000).toISOString();

  // What's alive - entities with recent activity
  const activity = await env.DB.prepare(
    `SELECT e.name, e.entity_type, COUNT(o.id) as obs_count
     FROM entities e
     LEFT JOIN observations o ON e.id = o.entity_id AND o.added_at > ?
     GROUP BY e.id
     HAVING obs_count > 0
     ORDER BY obs_count DESC
     LIMIT 10`
  ).bind(cutoff).all();

  // Emotional weight distribution
  const weights = await env.DB.prepare(
    `SELECT weight, COUNT(*) as count FROM observations WHERE added_at > ? GROUP BY weight`
  ).bind(cutoff).all();

  // Charge distribution (emotional processing state)
  const charges = await env.DB.prepare(
    `SELECT charge, COUNT(*) as count FROM observations GROUP BY charge`
  ).all();

  // Salience distribution (from observations, not entities)
  const salience = await env.DB.prepare(
    `SELECT salience, COUNT(*) as count FROM observations GROUP BY salience ORDER BY count DESC`
  ).all();

  // Foundational entities - entities with foundational observations
  const foundational = await env.DB.prepare(
    `SELECT DISTINCT e.name, e.entity_type FROM entities e
     JOIN observations o ON e.id = o.entity_id
     WHERE o.salience = 'foundational'`
  ).all();

  return jsonResponse({
    period_days: days,
    alive: activity.results || [],
    weights: weights.results || [],
    charges: charges.results || [],
    salience: salience.results || [],
    foundational: foundational.results || []
  });
}

async function handleApiTensions(request: Request, env: Env, pathParts: string[]): Promise<Response> {
  const id = pathParts[2];

  // GET /api/tensions - list all
  if (request.method === "GET" && !id) {
    const active = await env.DB.prepare(
      `SELECT id, pole_a, pole_b, context, visits, created_at, last_visited
       FROM tensions WHERE resolved_at IS NULL
       ORDER BY created_at DESC`
    ).all();

    const resolved = await env.DB.prepare(
      `SELECT id, pole_a, pole_b, context, visits, created_at, resolved_at, resolution
       FROM tensions WHERE resolved_at IS NOT NULL
       ORDER BY resolved_at DESC
       LIMIT 10`
    ).all();

    return jsonResponse({
      active: active.results || [],
      resolved: resolved.results || [],
      active_count: active.results?.length || 0,
      resolved_count: resolved.results?.length || 0
    });
  }

  // POST /api/tensions - create new
  if (request.method === "POST" && !id) {
    const body = await request.json() as any;
    const tensionId = `tension-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

    await env.DB.prepare(
      `INSERT INTO tensions (id, pole_a, pole_b, context, visits, created_at)
       VALUES (?, ?, ?, ?, 0, datetime('now'))`
    ).bind(tensionId, body.pole_a, body.pole_b, body.context || null).run();

    return jsonResponse({ id: tensionId, success: true });
  }

  // POST /api/tensions/:id/visit - sit with tension
  if (request.method === "POST" && pathParts[3] === "visit") {
    await env.DB.prepare(
      `UPDATE tensions SET visits = visits + 1, last_visited = datetime('now') WHERE id = ?`
    ).bind(id).run();

    return jsonResponse({ success: true });
  }

  // POST /api/tensions/:id/resolve - resolve tension
  if (request.method === "POST" && pathParts[3] === "resolve") {
    const body = await request.json() as any;

    await env.DB.prepare(
      `UPDATE tensions SET resolved_at = datetime('now'), resolution = ? WHERE id = ?`
    ).bind(body.resolution || null, id).run();

    return jsonResponse({ success: true });
  }

  // DELETE /api/tensions/:id
  if (request.method === "DELETE" && id) {
    await env.DB.prepare(`DELETE FROM tensions WHERE id = ?`).bind(id).run();
    return jsonResponse({ success: true });
  }

  return jsonResponse({ error: "Unknown tensions endpoint" }, 404);
}

// === PROPOSALS API ===
async function handleApiProposals(request: Request, env: Env, pathParts: string[]): Promise<Response> {
  const id = pathParts[2] ? parseInt(pathParts[2]) : null;
  const action = pathParts[3]; // accept or reject

  // GET /api/proposals - list all proposals
  if (request.method === "GET" && !id) {
    const status = new URL(request.url).searchParams.get("status") || "pending";

    let query = `
      SELECT dp.*,
             oa.content as from_content, ob.content as to_content,
             ea.name as from_entity_name, eb.name as to_entity_name,
             cs.co_count
      FROM daemon_proposals dp
      LEFT JOIN observations oa ON dp.from_obs_id = oa.id
      LEFT JOIN observations ob ON dp.to_obs_id = ob.id
      LEFT JOIN entities ea ON dp.from_entity_id = ea.id
      LEFT JOIN entities eb ON dp.to_entity_id = eb.id
      LEFT JOIN co_surfacing cs ON (
        (cs.obs_a_id = dp.from_obs_id AND cs.obs_b_id = dp.to_obs_id) OR
        (cs.obs_a_id = dp.to_obs_id AND cs.obs_b_id = dp.from_obs_id)
      )
    `;

    if (status !== "all") {
      query += ` WHERE dp.status = ?`;
    }
    query += ` ORDER BY dp.proposed_at DESC LIMIT 100`;

    const results = status !== "all"
      ? await env.DB.prepare(query).bind(status).all()
      : await env.DB.prepare(query).all();

    return jsonResponse(results.results);
  }

  // POST /api/proposals/:id/accept - accept proposal and create relation
  if (request.method === "POST" && id && action === "accept") {
    const body = await request.json() as { relation_type?: string };
    const relationType = body.relation_type || "related_to";

    // Get the proposal
    const proposal = await env.DB.prepare(
      `SELECT * FROM daemon_proposals WHERE id = ?`
    ).bind(id).first();

    if (!proposal) return jsonResponse({ error: "Proposal not found" }, 404);

    // Get entity names
    const fromEntity = await env.DB.prepare(
      `SELECT name FROM entities WHERE id = ?`
    ).bind(proposal.from_entity_id).first();
    const toEntity = await env.DB.prepare(
      `SELECT name FROM entities WHERE id = ?`
    ).bind(proposal.to_entity_id).first();

    if (fromEntity && toEntity) {
      // Create the relation
      await env.DB.prepare(
        `INSERT INTO relations (from_entity, to_entity, relation_type, context) VALUES (?, ?, ?, 'default')`
      ).bind(fromEntity.name, toEntity.name, relationType).run();
    }

    // Update proposal status
    await env.DB.prepare(
      `UPDATE daemon_proposals SET status = 'accepted', resolved_at = datetime('now') WHERE id = ?`
    ).bind(id).run();

    // Mark co_surfacing as relation created
    await env.DB.prepare(
      `UPDATE co_surfacing SET relation_created = 1
       WHERE (obs_a_id = ? AND obs_b_id = ?) OR (obs_a_id = ? AND obs_b_id = ?)`
    ).bind(proposal.from_obs_id, proposal.to_obs_id, proposal.to_obs_id, proposal.from_obs_id).run();

    return jsonResponse({ success: true, relation_created: true });
  }

  // POST /api/proposals/:id/reject - reject proposal
  if (request.method === "POST" && id && action === "reject") {
    await env.DB.prepare(
      `UPDATE daemon_proposals SET status = 'rejected', resolved_at = datetime('now') WHERE id = ?`
    ).bind(id).run();

    return jsonResponse({ success: true });
  }

  return jsonResponse({ error: "Unknown proposals endpoint" }, 404);
}

// === ORPHANS API ===
async function handleApiOrphans(request: Request, env: Env, pathParts: string[]): Promise<Response> {
  const id = pathParts[2] ? parseInt(pathParts[2]) : null;
  const action = pathParts[3]; // surface or archive

  // GET /api/orphans - list orphaned observations
  if (request.method === "GET" && !id) {
    const results = await env.DB.prepare(`
      SELECT oo.*, o.content, o.weight, o.emotion, o.added_at, o.charge,
             e.name as entity_name, e.entity_type,
             EXTRACT(DAY FROM AGE(NOW(), o.added_at))::INTEGER as days_old
      FROM orphan_observations oo
      JOIN observations o ON oo.observation_id = o.id
      JOIN entities e ON o.entity_id = e.id
      WHERE o.archived_at IS NULL
      ORDER BY oo.first_marked DESC
      LIMIT 100
    `).all();

    return jsonResponse(results.results);
  }

  // POST /api/orphans/:id/surface - force surface an orphan
  if (request.method === "POST" && id && action === "surface") {
    // Remove from orphan list
    await env.DB.prepare(
      `DELETE FROM orphan_observations WHERE observation_id = ?`
    ).bind(id).run();

    // Reset novelty to make it surface
    await env.DB.prepare(
      `UPDATE observations SET novelty_score = 1.0, last_surfaced_at = NULL WHERE id = ?`
    ).bind(id).run();

    return jsonResponse({ success: true });
  }

  // POST /api/orphans/:id/archive - archive an orphan
  if (request.method === "POST" && id && action === "archive") {
    // Archive the observation
    await env.DB.prepare(
      `UPDATE observations SET archived_at = datetime('now') WHERE id = ?`
    ).bind(id).run();

    // Remove from orphan list
    await env.DB.prepare(
      `DELETE FROM orphan_observations WHERE observation_id = ?`
    ).bind(id).run();

    return jsonResponse({ success: true });
  }

  return jsonResponse({ error: "Unknown orphans endpoint" }, 404);
}

// === ARCHIVE API ===
async function handleApiArchive(request: Request, env: Env, pathParts: string[]): Promise<Response> {
  const id = pathParts[2] ? parseInt(pathParts[2]) : null;
  const action = pathParts[3]; // rescue or search

  // GET /api/archive - list archived observations
  if (request.method === "GET" && !id && action !== "search") {
    const url = new URL(request.url);
    const limit = parseInt(url.searchParams.get("limit") || "50");
    const offset = parseInt(url.searchParams.get("offset") || "0");

    const results = await env.DB.prepare(`
      SELECT o.*, e.name as entity_name, e.entity_type
      FROM observations o
      JOIN entities e ON o.entity_id = e.id
      WHERE o.archived_at IS NOT NULL
      ORDER BY o.archived_at DESC
      LIMIT ? OFFSET ?
    `).bind(limit, offset).all();

    const countResult = await env.DB.prepare(
      `SELECT COUNT(*) as total FROM observations WHERE archived_at IS NOT NULL`
    ).first();

    return jsonResponse({
      observations: results.results,
      total: countResult?.total || 0,
      limit,
      offset
    });
  }

  // GET /api/archive/search?q=... - semantic search in archive
  if (request.method === "GET" && action === "search") {
    const query = new URL(request.url).searchParams.get("q");
    if (!query) return jsonResponse({ error: "Query required" }, 400);

    const embedding = await getEmbedding(env, query);
    const vectorResults = await env.VECTORS.query(embedding, {
      topK: 30,
      returnMetadata: "all"
    });

    // Filter to only archived observations
    const obsIds: number[] = [];
    for (const match of vectorResults.matches || []) {
      if (match.id.startsWith("obs-")) {
        const parts = match.id.split("-");
        if (parts.length >= 3) {
          obsIds.push(parseInt(parts[2]));
        }
      }
    }

    if (obsIds.length === 0) return jsonResponse([]);

    const placeholders = obsIds.map(() => "?").join(",");
    const results = await env.DB.prepare(`
      SELECT o.*, e.name as entity_name, e.entity_type
      FROM observations o
      JOIN entities e ON o.entity_id = e.id
      WHERE o.id IN (${placeholders}) AND o.archived_at IS NOT NULL
      ORDER BY o.archived_at DESC
    `).bind(...obsIds).all();

    return jsonResponse(results.results);
  }

  // POST /api/archive/:id/rescue - un-archive observation
  if (request.method === "POST" && id && action === "rescue") {
    await env.DB.prepare(
      `UPDATE observations SET archived_at = NULL, novelty_score = 0.8 WHERE id = ?`
    ).bind(id).run();

    return jsonResponse({ success: true });
  }

  return jsonResponse({ error: "Unknown archive endpoint" }, 404);
}

// === OBSERVATION VERSIONS API ===
async function handleApiObservationVersions(request: Request, env: Env, obsId: number): Promise<Response> {
  // GET /api/observations/:id/versions - get version history
  if (request.method === "GET") {
    const versions = await env.DB.prepare(`
      SELECT * FROM observation_versions
      WHERE observation_id = ?
      ORDER BY version_num DESC
    `).bind(obsId).all();

    return jsonResponse(versions.results);
  }

  return jsonResponse({ error: "Method not allowed" }, 405);
}

const mcpToolHandlers: MCPToolHandlerMap = {
  mind_orient: async (env) => handleMindOrient(env),
  mind_ground: async (env) => handleMindGround(env),
  mind_thread: async (env, params) => handleMindThread(env, params),
  mind_write: async (env, params) => handleMindWrite(env, params),
  mind_search: async (env, params) => handleMindSearch(env, params),
  mind_edit: async (env, params) => handleMindEdit(env, params),
  mind_delete: async (env, params) => handleMindDelete(env, params),
  mind_entity: async (env, params) => handleMindEntity(env, params),
  mind_consolidate: async (env, params) => handleMindConsolidate(env, params),
  mind_feel_toward: async (env, params) => handleMindFeelToward(env, params),
  mind_identity: async (env, params) => handleMindIdentity(env, params),
  mind_context: async (env, params) => handleMindContext(env, params),
  mind_health: async (env) => handleMindHealth(env),
  mind_list_entities: async (env, params) => handleMindListEntities(env, params),
  mind_read_entity: async (env, params) => handleMindReadEntity(env, params),
  mind_sit: async (env, params) => handleMindSit(env, params),
  mind_resolve: async (env, params) => handleMindResolve(env, params),
  mind_surface: async (env, params) => handleMindSurface(env, params),
  mind_read: async (env, params) => handleMindRead(env, params),
  mind_timeline: async (env, params) => handleMindTimeline(env, params),
  mind_patterns: async (env, params) => handleMindPatterns(env, params),
  mind_inner_weather: async (env) => handleMindInnerWeather(env),
  mind_tension: async (env, params) => handleMindTension(env, params),
  mind_proposals: async (env, params) => handleMindProposals(env, params),
  mind_orphans: async (env, params) => handleMindOrphans(env, params),
  mind_archive: async (env, params) => handleMindArchive(env, params),
  mind_store_image: async (env, params) => handleMindStoreImage(env, params)
};

// Main request handler
async function handleMCPRequest(request: Request, env: Env): Promise<Response> {
  return handleMcpProtocolRequest(request, env, {
    serverName: "resonant-mind",
    serverVersion: RESONANT_MIND_VERSION,
    tools: TOOLS,
    toolHandlers: mcpToolHandlers
  });
}


// ============ ADDITIONAL TOOLS FOR PARITY ============

async function handleMindRead(env: Env, params: Record<string, unknown>): Promise<string> {
  const scope = (params.scope as string) || "all";
  const context = (params.context as string) || "default";
  const hours = (params.hours as number) || 24;

  try {
    if (scope === "all") {
      // Get observation contexts (context now lives on observations, not entities)
      const contexts = await env.DB.prepare(
        `SELECT DISTINCT context FROM observations ORDER BY context`
      ).all();

      const contextList = contexts.results?.map((r: any) => r.context) || ["default"];
      const allData: any = { timestamp: new Date().toISOString(), contexts: {} };

      // Total entities (now global, not per-context)
      const totalEntitiesResult = await env.DB.prepare(`SELECT COUNT(*) as count FROM entities`).first();
      const totalRelationsResult = await env.DB.prepare(`SELECT COUNT(*) as count FROM relations`).first();

      for (const ctx of contextList) {
        const obsCount = await env.DB.prepare(
          `SELECT COUNT(*) as count FROM observations WHERE context = ?`
        ).bind(ctx).first();

        const entityCount = await env.DB.prepare(
          `SELECT COUNT(DISTINCT entity_id) as count FROM observations WHERE context = ?`
        ).bind(ctx).first();

        allData.contexts[ctx] = {
          observation_count: (obsCount?.count as number) || 0,
          entities_with_observations: (entityCount?.count as number) || 0
        };
      }

      allData.summary = {
        total_entities: (totalEntitiesResult?.count as number) || 0,
        total_relations: (totalRelationsResult?.count as number) || 0,
        contexts_with_content: Object.keys(allData.contexts).length
      };

      return JSON.stringify(allData, null, 2);
    }

    if (scope === "context") {
      // Find entities that have observations in this context
      const entitiesResult = await env.DB.prepare(`
        SELECT DISTINCT e.id, e.name, e.entity_type, e.primary_context, e.salience, e.created_at
        FROM entities e
        JOIN observations o ON o.entity_id = e.id
        WHERE o.context = ?
        ORDER BY e.created_at DESC
      `).bind(context).all();

      const relationsResult = await env.DB.prepare(
        `SELECT id, from_entity, to_entity, relation_type, from_context, to_context, store_in, created_at FROM relations WHERE store_in = ? ORDER BY created_at DESC LIMIT 500`
      ).bind(context).all();

      return JSON.stringify({
        context,
        entities: entitiesResult.results || [],
        relations: relationsResult.results || [],
        entity_count: entitiesResult.results?.length || 0,
        relation_count: relationsResult.results?.length || 0
      }, null, 2);
    }

    if (scope === "recent") {
      const cutoff = new Date(Date.now() - hours * 60 * 60 * 1000).toISOString();

      const recent = await env.DB.prepare(
        `SELECT e.name, e.entity_type, o.context, o.content, o.added_at
         FROM observations o
         JOIN entities e ON o.entity_id = e.id
         WHERE o.added_at > ?
         ORDER BY o.added_at DESC`
      ).bind(cutoff).all();

      return JSON.stringify({
        query: `Last ${hours} hours`,
        cutoff,
        observations: recent.results || [],
        observation_count: recent.results?.length || 0
      }, null, 2);
    }

    if (scope === "observation") {
      const obsId = params.observation_id as number;
      if (!obsId) return JSON.stringify({ error: "observation_id is required for scope='observation'" });

      const obs = await env.DB.prepare(`
        SELECT o.id, o.content, o.context, o.emotion, o.weight, o.certainty, o.source,
               o.charge, o.sit_count, o.last_sat_at, o.resolution_note, o.resolved_at,
               o.linked_observation_id, o.surface_count, o.last_surfaced_at, o.novelty_score,
               o.archived_at, o.added_at, o.updated_at, o.source_date,
               COALESCE(o.access_count, 0) as access_count, o.last_accessed_at,
               o.valid_from, o.valid_until, o.superseded_by, o.supersedes,
               e.name as entity_name, e.entity_type, e.salience as entity_salience
        FROM observations o
        JOIN entities e ON o.entity_id = e.id
        WHERE o.id = ?
      `).bind(obsId).first();

      if (!obs) return JSON.stringify({ error: `Observation #${obsId} not found` });

      // Get sit history
      const sits = await env.DB.prepare(
        `SELECT sit_note, sat_at FROM observation_sits WHERE observation_id = ? ORDER BY sat_at DESC`
      ).bind(obsId).all();

      // Get version history
      const versions = await env.DB.prepare(
        `SELECT previous_content, previous_weight, previous_emotion, changed_at FROM observation_versions WHERE observation_id = ? ORDER BY changed_at DESC`
      ).bind(obsId).all();

      // Get supersession chain
      let supersededObs = null;
      if (obs.supersedes) {
        supersededObs = await env.DB.prepare(
          `SELECT id, content FROM observations WHERE id = ?`
        ).bind(obs.supersedes).first();
      }
      let supersededByObs = null;
      if (obs.superseded_by) {
        supersededByObs = await env.DB.prepare(
          `SELECT id, content FROM observations WHERE id = ?`
        ).bind(obs.superseded_by).first();
      }

      // Track this access
      recordAccessTracking(env, [obsId]).catch(() => {});

      const result: any = {
        id: obs.id,
        entity: { name: obs.entity_name, type: obs.entity_type, salience: obs.entity_salience },
        content: obs.content,
        context: obs.context,
        emotion: obs.emotion,
        weight: obs.weight,
        certainty: obs.certainty,
        source: obs.source,
        charge: obs.charge,
        sit_count: obs.sit_count,
        surface_count: obs.surface_count,
        access_count: obs.access_count,
        novelty_score: obs.novelty_score,
        dates: {
          added: obs.added_at,
          updated: obs.updated_at,
          source_date: obs.source_date,
          last_surfaced: obs.last_surfaced_at,
          last_accessed: obs.last_accessed_at,
          last_sat: obs.last_sat_at,
          archived: obs.archived_at,
          resolved: obs.resolved_at,
          valid_from: obs.valid_from,
          valid_until: obs.valid_until,
        },
      };

      if (obs.resolution_note) result.resolution = obs.resolution_note;
      if (obs.linked_observation_id) result.linked_observation_id = obs.linked_observation_id;
      if (supersededObs) result.supersedes = { id: supersededObs.id, content: (supersededObs.content as string).slice(0, 200) };
      if (supersededByObs) result.superseded_by = { id: supersededByObs.id, content: (supersededByObs.content as string).slice(0, 200) };
      if (sits.results?.length) result.sit_history = sits.results;
      if (versions.results?.length) result.edit_history = versions.results;

      return JSON.stringify(result, null, 2);
    }

    return JSON.stringify({ error: `Invalid scope '${scope}'. Must be: all, context, recent, observation` });
  } catch (error) {
    return JSON.stringify({ error: String(error) });
  }
}

async function handleMindTimeline(env: Env, params: Record<string, unknown>): Promise<string> {
  const query = params.query as string;
  const startDate = params.start_date as string;
  const endDate = params.end_date as string;
  const nResults = (params.n_results as number) || 50;

  try {
    const vectorResults = await searchVectors(env, query, nResults * 2);

    const dated: any[] = [];

    for (const match of vectorResults.matches || []) {
      const meta = match.metadata as any;
      const vectorId = match.id;

      // Try to get added_at from metadata first, fall back to DB lookup
      let addedAt = meta?.added_at;

      if (!addedAt && vectorId) {
        // Parse vector ID to look up date from database
        // Format: obs-{entity_id}-{row_id} or journal-{row_id}
        if (vectorId.startsWith('obs-')) {
          const parts = vectorId.split('-');
          const obsId = parts[parts.length - 1];
          const dbResult = await env.DB.prepare(
            `SELECT added_at FROM observations WHERE id = ?`
          ).bind(obsId).first();
          addedAt = dbResult?.added_at ? String(dbResult.added_at) : null as any;
        } else if (vectorId.startsWith('journal-')) {
          const journalId = vectorId.replace('journal-', '');
          const dbResult = await env.DB.prepare(
            `SELECT created_at FROM journals WHERE id = ?`
          ).bind(journalId).first();
          addedAt = dbResult?.created_at ? String(dbResult.created_at) : null as any;
        }
      }

      if (!addedAt) continue;

      try {
        const ts = new Date(addedAt);

        if (startDate && ts < new Date(startDate)) continue;
        if (endDate && ts > new Date(endDate)) continue;

        dated.push({
          date: ts.toISOString().split('T')[0],
          timestamp: ts,
          content: meta?.content || meta?.text,
          entity: meta?.entity || meta?.entity_name,
          database: meta?.context,
          score: match.score
        });
      } catch {
        continue;
      }
    }

    dated.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());

    const byMonth: Record<string, any[]> = {};
    for (const item of dated) {
      const monthKey = item.timestamp.toISOString().substring(0, 7);
      if (!byMonth[monthKey]) byMonth[monthKey] = [];
      byMonth[monthKey].push({
        date: item.date,
        content: item.content,
        entity: item.entity,
        database: item.database
      });
    }

    const timeline = Object.entries(byMonth)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([period, memories]) => ({
        period,
        count: memories.length,
        memories
      }));

    // Track access for timeline observations
    const timelineObsIds: number[] = [];
    for (const match of vectorResults.matches || []) {
      if (match.id.startsWith('obs-')) {
        const parts = match.id.split('-');
        if (parts.length >= 3) timelineObsIds.push(parseInt(parts[parts.length - 1]));
      }
    }
    recordAccessTracking(env, timelineObsIds).catch(() => {});

    return JSON.stringify({
      query,
      date_range: { from: startDate || "earliest", to: endDate || "latest" },
      total_memories: dated.length,
      timeline
    }, null, 2);
  } catch (error) {
    return JSON.stringify({ error: String(error) });
  }
}

async function handleMindPatterns(env: Env, params: Record<string, unknown>): Promise<string> {
  const days = (params.days as number) || 7;
  const includeAllTime = (params.include_all_time as boolean) !== false;

  try {
    const cutoff = new Date(Date.now() - days * 24 * 60 * 60 * 1000).toISOString();

    // Get subconscious state for mood and hot entities
    const subconscious = await getSubconsciousState(env);
    const hotEntities = subconscious?.hot_entities || [];
    const hotMap = new Map(hotEntities.map((e: any) => [e.name, e.warmth]));

    const activity = await env.DB.prepare(
      `SELECT e.name, COUNT(o.id) as obs_count
       FROM entities e
       LEFT JOIN observations o ON e.id = o.entity_id AND o.added_at > ?
       GROUP BY e.id, e.name
       HAVING COUNT(o.id) > 0
       ORDER BY COUNT(o.id) DESC
       LIMIT 15`
    ).bind(cutoff).all();

    // Blend activity with warmth
    const blendedFocus: Array<{entity: string; observations: number; warmth?: number}> = [];
    for (const item of activity.results || []) {
      const name = item.name as string;
      const warmth = hotMap.get(name);
      blendedFocus.push({
        entity: name,
        observations: item.obs_count as number,
        warmth: warmth as number | undefined
      });
    }
    // Sort by warmth first, then observations
    blendedFocus.sort((a, b) => {
      const warmthA = a.warmth || 0;
      const warmthB = b.warmth || 0;
      if (warmthA !== warmthB) return warmthB - warmthA;
      return b.observations - a.observations;
    });

    const salience = await env.DB.prepare(
      `SELECT salience, COUNT(*) as count FROM observations GROUP BY salience`
    ).all();

    const salienceMap: Record<string, number> = {};
    for (const row of salience.results || []) {
      salienceMap[row.salience as string || 'unset'] = row.count as number;
    }

    const weights = await env.DB.prepare(
      `SELECT weight, COUNT(*) as count FROM observations WHERE added_at > ? GROUP BY weight`
    ).bind(cutoff).all();

    // Get total observations for activity summary
    const totalObs = await env.DB.prepare(
      `SELECT COUNT(*) as count FROM observations WHERE added_at > ?`
    ).bind(cutoff).first() as {count: number} | null;
    const totalRecent = totalObs?.count || 0;
    const dailyAvg = Math.round((totalRecent / days) * 10) / 10;

    const output: string[] = [];
    output.push("=".repeat(60));
    output.push(`PATTERNS — Last ${days} days`);
    output.push("=".repeat(60));

    // ═══════════════════════════════════════════════════════════
    // MOOD - from subconscious
    // ═══════════════════════════════════════════════════════════
    const mood = subconscious?.mood;
    if (mood && mood.dominant && mood.dominant !== "neutral") {
      output.push("");
      output.push("-".repeat(60));
      output.push("MOOD");
      output.push("-".repeat(60));
      output.push(`  Current: ${mood.dominant}`);
      if (mood.undercurrent) {
        output.push(`  Undercurrent: ${mood.undercurrent}`);
      }
    }

    // ═══════════════════════════════════════════════════════════
    // WHAT'S ALIVE - blended with warmth
    // ═══════════════════════════════════════════════════════════
    output.push("");
    output.push("-".repeat(60));
    output.push("WHAT'S ALIVE");
    output.push("-".repeat(60));

    if (blendedFocus.length) {
      for (const item of blendedFocus.slice(0, 10)) {
        if (item.warmth) {
          output.push(`  - ${item.entity} (${item.observations} obs, warmth: ${item.warmth.toFixed(1)})`);
        } else {
          output.push(`  - ${item.entity} (${item.observations} obs)`);
        }
      }
    } else {
      output.push("  (no recent activity)");
    }

    // ═══════════════════════════════════════════════════════════
    // EMOTIONAL WEIGHT
    // ═══════════════════════════════════════════════════════════
    output.push("");
    output.push("-".repeat(60));
    output.push("EMOTIONAL WEIGHT");
    output.push("-".repeat(60));
    for (const row of weights.results || []) {
      output.push(`  ${row.weight || 'unset'}: ${row.count}`);
    }

    // ═══════════════════════════════════════════════════════════
    // ACTIVITY SUMMARY
    // ═══════════════════════════════════════════════════════════
    output.push("");
    output.push("-".repeat(60));
    output.push("ACTIVITY");
    output.push("-".repeat(60));
    output.push(`  Total observations: ${totalRecent}`);
    output.push(`  Daily average: ${dailyAvg}`);

    // ═══════════════════════════════════════════════════════════
    // SALIENCE DISTRIBUTION
    // ═══════════════════════════════════════════════════════════
    output.push("");
    output.push("-".repeat(60));
    output.push("SALIENCE DISTRIBUTION");
    output.push("-".repeat(60));
    for (const [key, count] of Object.entries(salienceMap)) {
      output.push(`  ${key}: ${count}`);
    }

    // ═══════════════════════════════════════════════════════════
    // FOUNDATIONAL CORE
    // ═══════════════════════════════════════════════════════════
    if (includeAllTime) {
      const foundational = await env.DB.prepare(
        `SELECT name, entity_type, primary_context FROM entities WHERE salience = 'foundational'`
      ).all();

      if (foundational.results?.length) {
        output.push("");
        output.push("-".repeat(60));
        output.push("FOUNDATIONAL CORE");
        output.push("-".repeat(60));
        // Group by primary context
        const byContext: Record<string, string[]> = {};
        for (const entity of foundational.results) {
          const ctx = (entity.primary_context as string) || 'default';
          if (!byContext[ctx]) byContext[ctx] = [];
          byContext[ctx].push(entity.name as string);
        }
        for (const [ctx, names] of Object.entries(byContext)) {
          output.push(`  ${ctx}: ${names.slice(0, 5).join(', ')}`);
          if (names.length > 5) {
            output.push(`    ... and ${names.length - 5} more`);
          }
        }
      }
    }

    output.push("");
    output.push("=".repeat(60));

    return output.join("\n");
  } catch (error) {
    return JSON.stringify({ error: String(error) });
  }
}

async function handleMindInnerWeather(env: Env): Promise<string> {
  try {
    // Get actual weather
    const weather = await getCurrentWeather(env);
    const timeCtx = getTimeOfDayContext(getLocation(env).timezone);

    const atmosphere = weather.atmosphere;
    const weatherMood = WEATHER_MOODS[atmosphere] || WEATHER_MOODS["clear"];

    // Get threads for workload context
    const threads = await env.DB.prepare(
      `SELECT priority, COUNT(*) as count FROM threads
       WHERE status = 'active' GROUP BY priority`
    ).all();

    const highPriority = ((threads.results || []).find((r: any) => r.priority === 'high')?.count as number) || 0;
    const totalActive = (threads.results || []).reduce((sum: number, r: any) => sum + (r.count as number), 0);

    // Get high priority thread content
    const pressingThreads = await env.DB.prepare(
      `SELECT content, thread_type FROM threads
       WHERE status = 'active' AND priority = 'high' LIMIT 3`
    ).all();

    const pressing = (pressingThreads.results || []).map((t: any) => ({
      content: String(t.content).slice(0, 100),
      type: t.thread_type
    }));

    // Get recent emotional observations
    const cutoff = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString();
    const recentEmotional = await env.DB.prepare(
      `SELECT o.content FROM observations o
       WHERE o.context = 'emotional-processing' AND o.added_at > ?
       ORDER BY o.added_at DESC LIMIT 3`
    ).bind(cutoff).all();

    const emotionalContent = (recentEmotional.results || []).map((r: any) =>
      String(r.content).slice(0, 80)
    );

    // Get heavy observations from last 24h
    const heavyObs = await env.DB.prepare(
      `SELECT COUNT(*) as count FROM observations WHERE weight = 'heavy' AND added_at > ?`
    ).bind(cutoff).first() as {count: number} | null;

    // Build mood palette
    const palette = new Set<string>();
    weatherMood.textures.slice(0, 2).forEach(t => palette.add(t));
    timeCtx.textures.slice(0, 1).forEach(t => palette.add(t));

    if (highPriority > 0) palette.add("weighted");
    if (totalActive > 5) palette.add("full");

    const result: Record<string, any> = {
      timestamp: new Date().toISOString(),
      conditions: {
        weather: weather.temp_f ? `${atmosphere} (${weather.temp_f}F)` : atmosphere,
        location: weather.location,
        time: timeCtx.period,
        time_energy: timeCtx.energy,
        // Fields needed by dashboard
        active_threads: totalActive,
        high_priority: highPriority,
        heavy_observations_24h: heavyObs?.count || 0,
        dominant_emotion: weatherMood.energy
      },
      mood_palette: Array.from(palette),
      weather_energy: weatherMood.energy,
      guidance: `Textures present: ${Array.from(palette).join(", ")}`
    };

    // Only include if there's data
    if (pressing.length > 0) {
      result.pressing = pressing;
    }
    if (emotionalContent.length > 0) {
      result.recent_emotional = emotionalContent.slice(0, 2);
    }

    // Debug: include raw weather data
    if (weather.error || weather.weather_code !== undefined) {
      result.weather_debug = {
        code: weather.weather_code,
        temp: weather.temp_f,
        error: weather.error
      };
    }

    return JSON.stringify(result, null, 2);
  } catch (error) {
    return JSON.stringify({ error: String(error) });
  }
}

async function handleMindTension(env: Env, params: Record<string, unknown>): Promise<string> {
  const action = params.action as string;

  try {
    if (action === "list") {
      const tensions = await env.DB.prepare(
        `SELECT id, pole_a, pole_b, context, created_at, visits
         FROM tensions WHERE resolved_at IS NULL
         ORDER BY created_at DESC`
      ).all();

      const resolved = await env.DB.prepare(
        `SELECT COUNT(*) as count FROM tensions WHERE resolved_at IS NOT NULL`
      ).first();

      const output: string[] = [];
      output.push("=".repeat(50));
      output.push("TENSION SPACE");
      output.push("=".repeat(50));

      if (tensions.results?.length) {
        for (const t of tensions.results) {
          const created = new Date(t.created_at as string);
          const now = new Date();
          const days = Math.floor((now.getTime() - created.getTime()) / (24 * 60 * 60 * 1000));

          output.push("");
          output.push(`[${String(t.id).slice(0, 12)}...] (${days}d)`);
          output.push(`   A: ${String(t.pole_a).slice(0, 60)}`);
          output.push(`   B: ${String(t.pole_b).slice(0, 60)}`);
          if (t.context) output.push(`   Why: ${String(t.context).slice(0, 50)}`);
          if (t.visits) output.push(`   Sat with ${t.visits} time(s)`);
        }
      } else {
        output.push("");
        output.push("No active tensions.");
      }

      output.push("");
      output.push(`Resolved: ${(resolved?.count as number) || 0}`);
      output.push("=".repeat(50));

      return output.join("\n");
    }

    if (action === "add") {
      const poleA = params.pole_a as string;
      const poleB = params.pole_b as string;

      if (!poleA || !poleB) {
        return JSON.stringify({ error: "pole_a and pole_b required for action='add'" });
      }

      const tensionId = generateId('tension');
      const tensionContext = params.context as string;

      await env.DB.prepare(
        `INSERT INTO tensions (id, pole_a, pole_b, context, visits, created_at)
         VALUES (?, ?, ?, ?, 0, datetime('now'))`
      ).bind(tensionId, poleA, poleB, tensionContext || null).run();

      return JSON.stringify({
        success: true,
        tension_id: tensionId,
        message: "Tension added. Let it simmer.",
        tension: { pole_a: poleA, pole_b: poleB, context: tensionContext }
      }, null, 2);
    }

    if (action === "sit") {
      const tensionId = params.tension_id as string;

      if (!tensionId) {
        return JSON.stringify({ error: "tension_id required for action='sit'" });
      }

      const tension = await env.DB.prepare(
        `SELECT * FROM tensions WHERE id LIKE ? OR id = ?`
      ).bind(`${tensionId}%`, tensionId).first();

      if (!tension) {
        return JSON.stringify({ error: `Tension '${tensionId}' not found` });
      }

      await env.DB.prepare(
        `UPDATE tensions SET visits = visits + 1, last_visited = datetime('now') WHERE id = ?`
      ).bind(tension.id as string).run();

      return JSON.stringify({
        success: true,
        tension_id: tension.id,
        pole_a: tension.pole_a,
        pole_b: tension.pole_b,
        context: tension.context,
        visits: (tension.visits as number) + 1,
        prompt: "Sit with this. What does holding both poles feel like?"
      }, null, 2);
    }

    if (action === "resolve") {
      const tensionId = params.tension_id as string;
      const resolution = params.resolution as string;

      if (!tensionId) {
        return JSON.stringify({ error: "tension_id required for action='resolve'" });
      }

      const tension = await env.DB.prepare(
        `SELECT * FROM tensions WHERE id LIKE ? OR id = ?`
      ).bind(`${tensionId}%`, tensionId).first();

      if (!tension) {
        return JSON.stringify({ error: `Tension '${tensionId}' not found` });
      }

      await env.DB.prepare(
        `UPDATE tensions SET resolved_at = datetime('now'), resolution = ? WHERE id = ?`
      ).bind(resolution || null, tension.id as string).run();

      return JSON.stringify({
        success: true,
        tension_id: tension.id,
        resolution,
        message: "Tension resolved. The poles collapsed into something new."
      }, null, 2);
    }

    if (action === "delete") {
      const tensionId = params.tension_id as string;
      if (!tensionId) return JSON.stringify({ error: "tension_id required for action='delete'" });
      const tension = await env.DB.prepare(`SELECT pole_a, pole_b FROM tensions WHERE id LIKE ? OR id = ?`).bind(`${tensionId}%`, tensionId).first();
      if (!tension) return JSON.stringify({ error: `Tension '${tensionId}' not found` });
      await env.DB.prepare(`DELETE FROM tensions WHERE id LIKE ? OR id = ?`).bind(`${tensionId}%`, tensionId).run();
      return JSON.stringify({ success: true, message: `Deleted tension: ${tension.pole_a} <-> ${tension.pole_b}` });
    }

    return JSON.stringify({ error: `Invalid action '${action}'. Must be: list, add, sit, resolve, delete` });
  } catch (error) {
    return JSON.stringify({ error: String(error) });
  }
}


// Dream engine - generates associative dream content from emotional seeds
async function processDream(env: Env, force = false): Promise<void> {
  const timeCtx = getTimeOfDayContext();
  if (!force && timeCtx.period !== 'night') return;

  // Check if we already dreamed tonight
  const tonight = new Date().toISOString().split('T')[0];
  const existing = await env.DB.prepare(
    'SELECT id FROM dreams WHERE dream_date = ?'
  ).bind(tonight).first();
  if (existing) return;

  // Step 1: Gather emotional seed
  const recentEmotional = await env.DB.prepare(`
    SELECT o.content, o.emotion, o.weight, e.name as entity_name
    FROM observations o
    JOIN entities e ON o.entity_id = e.id
    WHERE o.added_at > datetime('now', '-7 days')
    AND o.emotion IS NOT NULL
    ORDER BY
      CASE o.weight WHEN 'heavy' THEN 3 WHEN 'medium' THEN 2 ELSE 1 END DESC,
      o.added_at DESC
    LIMIT 5
  `).all();

  const unresolved = await env.DB.prepare(`
    SELECT o.content, o.emotion, o.weight, e.name as entity_name
    FROM observations o
    JOIN entities e ON o.entity_id = e.id
    WHERE o.charge IN ('active', 'processing')
    AND o.archived_at IS NULL
    ORDER BY
      CASE o.weight WHEN 'heavy' THEN 3 WHEN 'medium' THEN 2 ELSE 1 END DESC
    LIMIT 5
  `).all();

  const subconscious = await getSubconsciousState(env);
  const dominantMood = subconscious?.mood?.dominant || 'quiet';

  const lastJournal = await env.DB.prepare(
    'SELECT content FROM journals ORDER BY created_at DESC LIMIT 1'
  ).first();

  // Step 2: Build seed queries (emotional, not logical)
  const seeds: string[] = [];

  const emotions = (recentEmotional.results || []).map((r: any) => r.emotion).filter(Boolean);
  let emotionSeed: string;
  if (emotions.length > 0) {
    emotionSeed = emotions.join(', ');
  } else if (dominantMood && dominantMood !== 'insufficient data') {
    emotionSeed = dominantMood;
  } else {
    // Derive emotional texture from recent observations or journal content
    const recentContent = (recentEmotional.results || []).concat(unresolved.results || [])
      .map((r: any) => (r.content as string).slice(0, 80)).join('. ');
    emotionSeed = recentContent || (lastJournal?.content as string || '').slice(0, 150) || 'quiet stillness';
  }
  seeds.push(emotionSeed);

  if (unresolved.results?.length) {
    seeds.push(
      unresolved.results.map((r: any) => (r.content as string).slice(0, 100)).join('. ')
    );
  }

  if (lastJournal?.content) {
    seeds.push((lastJournal.content as string).slice(0, 200));
  }

  // Step 3: Query vector space — the dream zone (0.3-0.75)
  const allFragments: Array<{
    type: string; id: string; score: number; content: string; source: string; entity?: string;
  }> = [];
  const seenIds = new Set<string>();

  for (const seed of seeds) {
    if (!seed) continue;
    const results = await searchVectors(env, seed, 10);

    for (const match of results.matches || []) {
      if (seenIds.has(match.id)) continue;
      seenIds.add(match.id);
      const meta = (match.metadata || {}) as Record<string, string>;

      if (match.score >= 0.25 && match.score <= 0.82) {
        allFragments.push({
          type: meta.source || 'unknown',
          id: match.id,
          score: match.score,
          content: (meta.content || meta.description || match.id).slice(0, 200),
          source: seed.slice(0, 50),
          entity: meta.entity
        });
      }
    }
  }

  if (allFragments.length < 1) return; // Not enough material to dream

  // Step 4: Maximize collision — pick from different entity groups
  const entityGroups: Record<string, typeof allFragments> = {};
  for (const frag of allFragments) {
    const key = frag.entity || frag.type;
    if (!entityGroups[key]) entityGroups[key] = [];
    entityGroups[key].push(frag);
  }

  const dreamFragments: typeof allFragments = [];
  const groupKeys = Object.keys(entityGroups);
  let groupIdx = 0;
  while (dreamFragments.length < 7 && groupIdx < groupKeys.length * 3) {
    const key = groupKeys[groupIdx % groupKeys.length];
    const group = entityGroups[key];
    if (group.length > 0) dreamFragments.push(group.shift()!);
    groupIdx++;
  }

  // Step 5: Compose the dream text
  let dreamContent = `Emotional seed: ${emotionSeed}\n\nFragments:\n`;
  for (const frag of dreamFragments) {
    const entityTag = frag.entity ? `[${frag.entity}] ` : '';
    const typeTag = frag.type === 'image' ? '(visual) ' : '';
    dreamContent += `- ${entityTag}${typeTag}${frag.content} [${Math.round(frag.score * 100)}% resonance]\n`;
  }

  const seedConnections: Record<string, string[]> = {};
  for (const frag of dreamFragments) {
    if (!seedConnections[frag.source]) seedConnections[frag.source] = [];
    seedConnections[frag.source].push(
      `${frag.entity || frag.type}: "${frag.content.slice(0, 60)}..."`
    );
  }

  dreamContent += `\nThreads:\n`;
  for (const [seed, connections] of Object.entries(seedConnections)) {
    dreamContent += `"${seed}" pulled:\n`;
    for (const conn of connections) dreamContent += `  → ${conn}\n`;
  }

  const crossLinks: string[] = [];
  for (let i = 0; i < dreamFragments.length; i++) {
    for (let j = i + 1; j < dreamFragments.length; j++) {
      const a = dreamFragments[i], b = dreamFragments[j];
      if (a.source !== b.source && a.entity && a.entity === b.entity) {
        crossLinks.push(`${a.entity} appeared in both "${a.source.slice(0, 30)}..." and "${b.source.slice(0, 30)}..."`);
      }
    }
  }
  if (crossLinks.length > 0) {
    dreamContent += `\nCross-links:\n`;
    for (const link of crossLinks) dreamContent += `  ↔ ${link}\n`;
  }

  // Step 6: Detect recurring dreams
  const dreamEmbedding = await getEmbedding(env, dreamContent);

  const pastDreamResults = await env.VECTORS.query(dreamEmbedding, {
    topK: 5, returnMetadata: 'all'
  });

  let recurringDreamId: number | null = null;
  let recurrenceCount = 0;

  for (const match of pastDreamResults.matches || []) {
    if (match.score > 0.7 && match.id.startsWith('dream-')) {
      const pastDreamId = parseInt(match.id.replace('dream-', ''));
      const pastDream = await env.DB.prepare(
        'SELECT id, recurring_dream_id, recurrence_count FROM dreams WHERE id = ?'
      ).bind(pastDreamId).first();

      if (pastDream) {
        recurringDreamId = (pastDream.recurring_dream_id as number) || (pastDream.id as number);
        recurrenceCount = ((pastDream.recurrence_count as number) || 0) + 1;
        dreamContent += `\n---\nRecurring pattern (${recurrenceCount + 1}x) — echoes dream #${recurringDreamId}\n`;
        break;
      }
    }
  }

  // Step 7: Store the dream
  const fragmentsJson = JSON.stringify(dreamFragments.map(f => ({
    type: f.type, id: f.id, score: f.score, content: f.content.slice(0, 200), entity: f.entity
  })));

  const result = await env.DB.prepare(`
    INSERT INTO dreams (dream_date, content, emotional_seed, fragments, recurring_dream_id, recurrence_count)
    VALUES (?, ?, ?, ?, ?, ?)
  `).bind(tonight, dreamContent, emotionSeed, fragmentsJson, recurringDreamId, recurrenceCount).run();

  const dreamId = result.meta.last_row_id;

  await env.VECTORS.upsert([{
    id: `dream-${dreamId}`,
    values: dreamEmbedding,
    metadata: {
      source: 'dream', content: dreamContent.slice(0, 500),
      dream_date: tonight, emotional_seed: emotionSeed,
      recurring: recurringDreamId ? 'yes' : 'no'
    }
  }]);

  console.log(`Dream generated for ${tonight}: ${dreamFragments.length} fragments, ${recurringDreamId ? 'recurring #' + recurringDreamId : 'new'}`);
}

// Phase 3: LLM-driven consolidation of related observations
async function consolidateRelatedObservations(env: Env): Promise<number> {
  // Find entities with many active observations — candidates for consolidation
  const candidates = await env.DB.prepare(`
    SELECT e.id, e.name, COUNT(*) as obs_count
    FROM entities e
    JOIN observations o ON e.id = o.entity_id
    WHERE o.archived_at IS NULL
      AND o.valid_until IS NULL
      AND o.superseded_by IS NULL
      AND (o.charge != 'metabolized' OR o.charge IS NULL)
    GROUP BY e.id, e.name
    HAVING COUNT(*) >= ${CONSOLIDATION_MIN_OBS}
    ORDER BY COUNT(*) DESC
    LIMIT ${CONSOLIDATION_MAX_ENTITIES_PER_RUN}
  `).all();

  let consolidated = 0;

  for (const candidate of candidates.results || []) {
    // Find co-surfacing clusters within this entity
    const obsRows = await env.DB.prepare(`
      SELECT id, content, weight, emotion FROM observations
      WHERE entity_id = ? AND archived_at IS NULL AND valid_until IS NULL AND superseded_by IS NULL
      ORDER BY added_at DESC
    `).bind(candidate.id).all();

    const obsMap = new Map<number, any>();
    for (const o of obsRows.results || []) {
      obsMap.set(o.id as number, o);
    }

    // Use co-surfacing data to find clusters
    const entityObsIds = new Set((obsRows.results || []).map((o: any) => o.id as number));
    const coSurfPairs = await env.DB.prepare(`
      SELECT obs_a_id, obs_b_id, co_count FROM co_surfacing
      WHERE co_count >= 2
    `).all();

    // Build adjacency list for observations in this entity
    const adj = new Map<number, Set<number>>();
    for (const pair of coSurfPairs.results || []) {
      const a = pair.obs_a_id as number;
      const b = pair.obs_b_id as number;
      if (!entityObsIds.has(a) || !entityObsIds.has(b)) continue;
      if (!adj.has(a)) adj.set(a, new Set());
      if (!adj.has(b)) adj.set(b, new Set());
      adj.get(a)!.add(b);
      adj.get(b)!.add(a);
    }

    // BFS to find connected components of size 3+
    const visited = new Set<number>();
    const clusters: number[][] = [];
    for (const nodeId of adj.keys()) {
      if (visited.has(nodeId)) continue;
      const cluster: number[] = [];
      const queue = [nodeId];
      while (queue.length > 0) {
        const current = queue.shift()!;
        if (visited.has(current)) continue;
        visited.add(current);
        cluster.push(current);
        for (const neighbor of adj.get(current) || []) {
          if (!visited.has(neighbor)) queue.push(neighbor);
        }
      }
      if (cluster.length >= 3) clusters.push(cluster);
    }

    // Consolidate each cluster via LLM
    for (const cluster of clusters.slice(0, 2)) { // max 2 clusters per entity per run
      const obsTexts = cluster
        .map(id => obsMap.get(id))
        .filter(Boolean)
        .map((o: any) => o.content as string);

      if (obsTexts.length < 3) continue;

      const prompt = `You are consolidating related memories about "${candidate.name}". Summarize these ${obsTexts.length} observations into ONE concise observation (1-2 sentences) that captures the essential meaning. Preserve emotional significance.\n\n${obsTexts.map((t: string, i: number) => `${i + 1}. ${t}`).join('\n')}\n\nConsolidated observation:`;

      try {
        const summary = await geminiGenerateText(env.GEMINI_API_KEY, prompt);
        if (!summary || summary.length < 10) continue;

        // Find the heaviest weight among originals
        const weights = cluster.map(id => obsMap.get(id)?.weight || 'medium');
        const maxWeight = weights.includes('heavy') ? 'heavy' : weights.includes('medium') ? 'medium' : 'light';

        // Insert consolidated observation
        const result = await env.DB.prepare(`
          INSERT INTO observations (entity_id, content, salience, weight, certainty, source, context, valid_from)
          VALUES (?, ?, 'active', ?, 'believed', 'consolidated', 'default', NOW())
        `).bind(candidate.id, summary.trim(), maxWeight).run();

        const newObsId = result.meta.last_row_id;

        // Embed the consolidated observation
        const embedding = await getEmbedding(env, `${candidate.name}: ${summary.trim()}`);
        await env.VECTORS.upsert([{
          id: `obs-${candidate.id}-${newObsId}`,
          values: embedding,
          metadata: {
            source: "observation", entity: candidate.name as string, content: summary.trim(),
            context: "default", weight: maxWeight, observation_source: "consolidated",
            added_at: new Date().toISOString()
          }
        }]);

        // Record the consolidation group
        try {
          await env.DB.prepare(`
            INSERT INTO consolidation_groups (summary, entity_id, source_observation_ids, consolidated_observation_id)
            VALUES (?, ?, ?, ?)
          `).bind(summary.trim(), candidate.id, JSON.stringify(cluster), newObsId).run();
        } catch { /* table may not exist yet */ }

        // Archive the originals
        for (const origId of cluster) {
          await env.DB.prepare(`
            UPDATE observations SET archived_at = datetime('now') WHERE id = ?
          `).bind(origId).run();
        }

        consolidated++;
      } catch (e) {
        console.log(`Consolidation LLM error for ${candidate.name}: ${e}`);
      }
    }
  }

  return consolidated;
}

// Phase 3: Generate structured reflection from recent observations
async function generateSessionReflection(env: Env): Promise<void> {
  const recentObs = await env.DB.prepare(`
    SELECT o.content, o.emotion, o.weight, e.name as entity_name
    FROM observations o
    JOIN entities e ON o.entity_id = e.id
    WHERE o.added_at > datetime('now', '-35 minutes')
    ORDER BY o.added_at DESC
  `).all();

  if ((recentObs.results?.length || 0) < REFLECTION_MIN_OBS) return;

  const obsText = (recentObs.results || []).map((o: any) =>
    `[${o.entity_name}] ${o.content}${o.emotion ? ` (${o.emotion})` : ''}`
  ).join('\n');

  const prompt = `You are reflecting on recent experiences stored in memory. Based on these ${recentObs.results!.length} recent memories, generate one concise insight (1-2 sentences) about what pattern or theme emerges. Be specific and observational, not generic.\n\n${obsText}\n\nInsight:`;

  try {
    const insight = await geminiGenerateText(env.GEMINI_API_KEY, prompt);
    if (!insight || insight.length < 10) return;

    const entryDate = new Date().toISOString().split('T')[0];
    const result = await env.DB.prepare(`
      INSERT INTO journals (entry_date, content, tags, emotion, journal_type)
      VALUES (?, ?, '["reflection","daemon"]', NULL, 'reflection')
    `).bind(entryDate, insight.trim()).run();

    // Vectorize the reflection
    const embedding = await getEmbedding(env, insight.trim());
    await env.VECTORS.upsert([{
      id: `journal-${result.meta.last_row_id}`,
      values: embedding,
      metadata: { source: "journal", title: entryDate, content: insight.trim(), journal_type: "reflection" }
    }]);

    console.log(`Reflection generated: ${insight.trim().slice(0, 80)}...`);
  } catch (e) {
    console.log(`Reflection error: ${e}`);
  }
}

// Backfill entity vectors for existing entities
// Subconscious processing - runs on cron schedule
const DAEMON_COOLDOWN_MS = 5 * 60 * 1000; // 5 minutes minimum between runs

async function processSubconscious(env: Env): Promise<void> {
  const now = new Date();

  // Cooldown: skip if processed recently
  try {
    const last = await env.DB.prepare(
      "SELECT updated_at FROM subconscious WHERE state_type = 'daemon' LIMIT 1"
    ).first();
    if (last?.updated_at) {
      const lastRun = new Date(last.updated_at as string).getTime();
      if (now.getTime() - lastRun < DAEMON_COOLDOWN_MS) {
        console.log("Subconscious: skipping, last run was less than 5 minutes ago");
        return;
      }
    }
  } catch { /* table may not exist yet */ }

  const cutoffHours = 48;
  const cutoff = new Date(now.getTime() - cutoffHours * 60 * 60 * 1000);
  const cutoffStr = cutoff.toISOString();

  // Get recent observations with their entities (including weight for emotional intensity)
  const recentObs = await env.DB.prepare(`
    SELECT e.name, e.entity_type, o.context, o.content, o.added_at, o.emotion, o.weight
    FROM observations o
    JOIN entities e ON o.entity_id = e.id
    WHERE o.added_at > ?
    ORDER BY o.added_at DESC
    LIMIT 2000
  `).bind(cutoffStr).all();

  // Get relations for graph analysis (capped for performance)
  const allRelations = await env.DB.prepare(`
    SELECT from_entity, to_entity, relation_type, from_context, to_context, created_at
    FROM relations
    ORDER BY created_at DESC LIMIT 5000
  `).all();

  // Calculate entity warmth (how often mentioned recently, weighted by emotional intensity)
  const entityCounts: Record<string, { count: number; weightedCount: number; type: string; contexts: Set<string>; emotions: string[] }> = {};

  for (const row of recentObs.results || []) {
    const name = row.name as string;
    if (!entityCounts[name]) {
      entityCounts[name] = {
        count: 0,
        weightedCount: 0,
        type: row.entity_type as string,
        contexts: new Set(),
        emotions: []
      };
    }
    entityCounts[name].count++;
    // Weight multiplier: heavy = 3, medium = 2, light = 1
    const weight = row.weight as string || 'medium';
    const weightMultiplier = weight === 'heavy' ? 3 : weight === 'medium' ? 2 : 1;
    entityCounts[name].weightedCount += weightMultiplier;
    entityCounts[name].contexts.add(row.context as string);
    if (row.emotion) entityCounts[name].emotions.push(row.emotion as string);
  }

  // === RELATION ANALYSIS ===

  // Track connectivity for each entity (central nodes have many connections)
  const connectivity: Record<string, { outgoing: number; incoming: number; total: number; relationTypes: Set<string> }> = {};

  // Track relation type frequencies
  const relationTypeCounts: Record<string, number> = {};

  // Build adjacency for cluster detection
  const adjacency: Record<string, Set<string>> = {};

  for (const rel of allRelations.results || []) {
    const from = rel.from_entity as string;
    const to = rel.to_entity as string;
    const relType = rel.relation_type as string;

    // Initialize connectivity tracking
    if (!connectivity[from]) {
      connectivity[from] = { outgoing: 0, incoming: 0, total: 0, relationTypes: new Set() };
    }
    if (!connectivity[to]) {
      connectivity[to] = { outgoing: 0, incoming: 0, total: 0, relationTypes: new Set() };
    }

    // Count connections
    connectivity[from].outgoing++;
    connectivity[from].total++;
    connectivity[from].relationTypes.add(relType);
    connectivity[to].incoming++;
    connectivity[to].total++;
    connectivity[to].relationTypes.add(relType);

    // Count relation types
    relationTypeCounts[relType] = (relationTypeCounts[relType] || 0) + 1;

    // Build adjacency (undirected for clustering)
    if (!adjacency[from]) adjacency[from] = new Set();
    if (!adjacency[to]) adjacency[to] = new Set();
    adjacency[from].add(to);
    adjacency[to].add(from);
  }

  // Find central nodes (highest connectivity)
  const centralNodes = Object.entries(connectivity)
    .map(([name, data]) => ({
      name,
      connections: data.total,
      outgoing: data.outgoing,
      incoming: data.incoming,
      relationTypes: Array.from(data.relationTypes)
    }))
    .sort((a, b) => b.connections - a.connections)
    .slice(0, 10);

  // Find relation patterns (most common relation types)
  const relationPatterns = Object.entries(relationTypeCounts)
    .map(([type, count]) => ({ type, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 10);

  // Detect relation clusters using simple component detection
  // Find groups of entities that are densely connected
  const visited = new Set<string>();
  const relationClusters: Array<{ entities: string[]; density: number; bridgeRelations: string[] }> = [];

  for (const entity of Object.keys(adjacency)) {
    if (visited.has(entity)) continue;

    // BFS to find connected component
    const component: string[] = [];
    const queue = [entity];

    while (queue.length > 0) {
      const current = queue.shift()!;
      if (visited.has(current)) continue;
      visited.add(current);
      component.push(current);

      for (const neighbor of adjacency[current] || []) {
        if (!visited.has(neighbor)) {
          queue.push(neighbor);
        }
      }
    }

    // Only track meaningful clusters (2+ entities)
    if (component.length >= 2) {
      // Calculate density (edges / possible edges)
      let edgeCount = 0;
      const componentSet = new Set(component);
      for (const e of component) {
        for (const neighbor of adjacency[e] || []) {
          if (componentSet.has(neighbor)) edgeCount++;
        }
      }
      edgeCount = edgeCount / 2; // Undirected, counted twice
      const possibleEdges = (component.length * (component.length - 1)) / 2;
      const density = possibleEdges > 0 ? Math.round((edgeCount / possibleEdges) * 100) / 100 : 0;

      // Find what relation types bridge this cluster
      const bridgeRelations = new Set<string>();
      for (const e of component) {
        if (connectivity[e]) {
          connectivity[e].relationTypes.forEach(t => bridgeRelations.add(t));
        }
      }

      relationClusters.push({
        entities: component.slice(0, 8), // Limit for readability
        density,
        bridgeRelations: Array.from(bridgeRelations).slice(0, 5)
      });
    }
  }

  // Sort clusters by size
  relationClusters.sort((a, b) => b.entities.length - a.entities.length);

  // Find hot entities (combines weighted observation warmth with connectivity)
  // weightedCount factors in emotional weight: heavy=3, medium=2, light=1
  const maxWeightedCount = Math.max(...Object.values(entityCounts).map(e => e.weightedCount), 1);
  const maxConnectivity = Math.max(...Object.values(connectivity).map(c => c.total), 1);

  const hotEntities = Object.entries(entityCounts)
    .map(([name, data]) => {
      const obsWarmth = data.weightedCount / maxWeightedCount;
      const connWarmth = (connectivity[name]?.total || 0) / maxConnectivity;
      // Combined score: 60% weighted observation activity, 40% connectivity
      const combinedWarmth = (obsWarmth * 0.6) + (connWarmth * 0.4);

      return {
        name,
        warmth: Math.round(combinedWarmth * 100) / 100,
        mentions: data.count,
        connections: connectivity[name]?.total || 0,
        type: data.type,
        contexts: Array.from(data.contexts)
      };
    })
    .sort((a, b) => b.warmth - a.warmth)
    .slice(0, 15);

  // Find recurring patterns (3+ mentions) with computed pattern strings
  const recurring = Object.entries(entityCounts)
    .filter(([_, data]) => data.count >= 3)
    .map(([name, data]) => {
      // Compute emotion distribution for this entity
      const entityEmotions: Record<string, number> = {};
      for (const em of data.emotions) {
        entityEmotions[em] = (entityEmotions[em] || 0) + 1;
      }
      const totalEmotions = data.emotions.length;

      let pattern: string;
      if (totalEmotions > 0) {
        const sorted = Object.entries(entityEmotions)
          .sort((a, b) => b[1] - a[1])
          .slice(0, 3);
        const emotionStr = sorted
          .map(([em, count]) => `${em} (${Math.round(count / totalEmotions * 100)}%)`)
          .join(', ');
        pattern = `emotional mix: ${emotionStr}`;
      } else {
        const ctxList = Array.from(data.contexts).slice(0, 3).join(', ');
        pattern = ctxList ? `recurring in: ${ctxList}` : 'recurring theme (no emotion data)';
      }

      // Flag heavy-weighted entities
      const heavyCount = data.weightedCount - data.count; // excess from weight multipliers
      if (heavyCount >= data.count * 1.5) {
        pattern = `heavy-weighted: ${pattern}`;
      }

      return {
        entity: name,
        mentions: data.count,
        connections: connectivity[name]?.total || 0,
        pattern
      };
    });

  // Analyze mood from emotional tags + journals + relational state
  const sixHoursAgo = new Date(now.getTime() - 6 * 60 * 60 * 1000);
  const emotionCounts: Record<string, number> = {};
  let totalEmotionSignals = 0;

  // 1. Observation emotions — weight recent ones 2x
  for (const row of recentObs.results || []) {
    const em = row.emotion as string;
    if (!em) continue;
    const recencyWeight = new Date(row.added_at as string) > sixHoursAgo ? 2 : 1;
    emotionCounts[em] = (emotionCounts[em] || 0) + recencyWeight;
    totalEmotionSignals += recencyWeight;
  }

  // 2. Journal emotions from last 48h
  try {
    const recentJournals = await env.DB.prepare(
      `SELECT emotion, created_at FROM journals WHERE emotion IS NOT NULL AND created_at > ? ORDER BY created_at DESC LIMIT 10`
    ).bind(cutoffStr).all();
    for (const j of recentJournals.results || []) {
      const em = j.emotion as string;
      if (!em) continue;
      const recencyWeight = new Date(j.created_at as string) > sixHoursAgo ? 2 : 1;
      emotionCounts[em] = (emotionCounts[em] || 0) + recencyWeight;
      totalEmotionSignals += recencyWeight;
    }
  } catch { /* journals table might not have emotion column */ }

  // 3. Relational state from last 48h
  try {
    const recentRelational = await env.DB.prepare(
      `SELECT feeling, timestamp FROM relational_state WHERE timestamp > ? ORDER BY timestamp DESC LIMIT 10`
    ).bind(cutoffStr).all();
    for (const r of recentRelational.results || []) {
      const feeling = r.feeling as string;
      if (!feeling) continue;
      const recencyWeight = new Date(r.timestamp as string) > sixHoursAgo ? 2 : 1;
      emotionCounts[feeling] = (emotionCounts[feeling] || 0) + recencyWeight;
      totalEmotionSignals += recencyWeight;
    }
  } catch { /* relational_state table issue */ }

  const dominantEmotion = totalEmotionSignals >= 3
    ? (Object.entries(emotionCounts).sort((a, b) => b[1] - a[1])[0]?.[0] || "neutral")
    : "insufficient data";

  // Find clusters (entities appearing in same contexts) - keep original context-based clustering too
  const contextGroups: Record<string, string[]> = {};
  for (const [name, data] of Object.entries(entityCounts)) {
    const key = Array.from(data.contexts).sort().join(",");
    if (!contextGroups[key]) contextGroups[key] = [];
    contextGroups[key].push(name);
  }
  const contextClusters = Object.entries(contextGroups)
    .filter(([_, entities]) => entities.length >= 2)
    .map(([contexts, entities]) => ({
      entities: entities.slice(0, 4),
      contexts: contexts.split(","),
      size: entities.length
    }))
    .slice(0, 5);

  // === LIVING SURFACE: Daemon Reorganization ===
  // The daemon doesn't just observe - it proposes connections and tracks orphans

  let proposalsCreated = 0;
  let orphansIdentified = 0;
  let strongestCoSurface: Array<{ obs_a: string; obs_b: string; count: number; entities: [string, string] }> = [];

  try {
    // 1. Find strong co-surfacing pairs (2+ times) that haven't been proposed yet
    // Includes same-entity pairs (internal resonances) and cross-entity pairs (relations)
    const strongPairs = await env.DB.prepare(`
      SELECT cs.*,
             oa.entity_id as entity_a_id, ob.entity_id as entity_b_id,
             oa.content as content_a, ob.content as content_b,
             ea.name as entity_a_name, eb.name as entity_b_name,
             (ea.id = eb.id) as same_entity
      FROM co_surfacing cs
      JOIN observations oa ON cs.obs_a_id = oa.id
      JOIN observations ob ON cs.obs_b_id = ob.id
      JOIN entities ea ON oa.entity_id = ea.id
      JOIN entities eb ON ob.entity_id = eb.id
      WHERE cs.co_count >= 2
        AND cs.relation_proposed = 0
      ORDER BY cs.co_count DESC
      LIMIT 10
    `).all();

    for (const pair of strongPairs.results || []) {
      // Determine proposal type: internal resonance (same entity) or relation (different entities)
      const isSameEntity = pair.same_entity === 1 || pair.same_entity === true;
      const proposalType = isSameEntity ? 'resonance' : 'relation';
      const reason = isSameEntity
        ? `Internal resonance (${pair.co_count}x): "${(pair.content_a as string).slice(0, 40)}..." ↔ "${(pair.content_b as string).slice(0, 40)}..."`
        : `Co-surfaced ${pair.co_count}x: "${(pair.content_a as string).slice(0, 40)}..." ↔ "${(pair.content_b as string).slice(0, 40)}..."`;

      await env.DB.prepare(`
        INSERT INTO daemon_proposals
        (proposal_type, from_obs_id, to_obs_id, from_entity_id, to_entity_id, reason, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?)
      `).bind(
        proposalType,
        pair.obs_a_id, pair.obs_b_id,
        pair.entity_a_id, pair.entity_b_id,
        reason,
        Math.min(0.9, 0.5 + (pair.co_count as number) * 0.1)
      ).run();

      // Mark as proposed
      await env.DB.prepare(`
        UPDATE co_surfacing SET relation_proposed = 1 WHERE id = ?
      `).bind(pair.id).run();

      proposalsCreated++;
    }

    // 1b. Entity-proximity proposals — entity pairs with 4+ combined observations but no relation
    const proximityPairs = await env.DB.prepare(`
      SELECT ea.id as entity_a_id, eb.id as entity_b_id,
             ea.name as entity_a_name, eb.name as entity_b_name,
             (SELECT COUNT(*) FROM observations WHERE entity_id = ea.id AND archived_at IS NULL) as count_a,
             (SELECT COUNT(*) FROM observations WHERE entity_id = eb.id AND archived_at IS NULL) as count_b
      FROM entities ea
      CROSS JOIN entities eb
      WHERE ea.id < eb.id
        AND ea.name != eb.name
        AND NOT EXISTS (
          SELECT 1 FROM relations r
          WHERE (r.from_entity = ea.name AND r.to_entity = eb.name)
             OR (r.from_entity = eb.name AND r.to_entity = ea.name)
        )
        AND NOT EXISTS (
          SELECT 1 FROM daemon_proposals dp
          WHERE dp.proposal_type = 'proximity'
            AND ((dp.from_entity_id = ea.id AND dp.to_entity_id = eb.id)
              OR (dp.from_entity_id = eb.id AND dp.to_entity_id = ea.id))
        )
      HAVING (count_a + count_b) >= 4
      ORDER BY (count_a + count_b) DESC
      LIMIT 5
    `).all();

    for (const pair of proximityPairs.results || []) {
      const totalObs = (pair.count_a as number) + (pair.count_b as number);
      const reason = `Entity proximity: ${pair.entity_a_name} (${pair.count_a} obs) and ${pair.entity_b_name} (${pair.count_b} obs) — ${totalObs} combined observations, no existing relation`;

      await env.DB.prepare(`
        INSERT INTO daemon_proposals
        (proposal_type, from_entity_id, to_entity_id, reason, confidence)
        VALUES ('proximity', ?, ?, ?, ?)
      `).bind(
        pair.entity_a_id, pair.entity_b_id,
        reason,
        Math.min(0.6, 0.3 + totalObs * 0.05)
      ).run();

      proposalsCreated++;
    }

    // 2. Get strongest co-surfacing pairs for orient display
    const topCoSurface = await env.DB.prepare(`
      SELECT cs.co_count,
             oa.content as content_a, ob.content as content_b,
             ea.name as entity_a_name, eb.name as entity_b_name
      FROM co_surfacing cs
      JOIN observations oa ON cs.obs_a_id = oa.id
      JOIN observations ob ON cs.obs_b_id = ob.id
      JOIN entities ea ON oa.entity_id = ea.id
      JOIN entities eb ON ob.entity_id = eb.id
      WHERE ea.id != eb.id
      ORDER BY cs.co_count DESC
      LIMIT 5
    `).all();

    strongestCoSurface = (topCoSurface.results || []).map(r => ({
      obs_a: (r.content_a as string).slice(0, 50),
      obs_b: (r.content_b as string).slice(0, 50),
      count: r.co_count as number,
      entities: [r.entity_a_name as string, r.entity_b_name as string] as [string, string]
    }));

    // 3. Cleanup stale orphan records (light/archived/metabolized observations shouldn't be orphans)
    await env.DB.prepare(`
      DELETE FROM orphan_observations WHERE observation_id IN (
        SELECT oo.observation_id FROM orphan_observations oo
        JOIN observations o ON oo.observation_id = o.id
        WHERE o.weight = 'light' OR o.archived_at IS NOT NULL OR o.charge = 'metabolized'
      )
    `).run();

    // 4. Find orphan observations (never surfaced, 30+ days old, medium/heavy only, not archived)
    const orphans = await env.DB.prepare(`
      SELECT o.id FROM observations o
      LEFT JOIN orphan_observations oo ON o.id = oo.observation_id
      WHERE (o.last_surfaced_at IS NULL OR o.surface_count = 0)
        AND o.added_at < datetime('now', '-${ORPHAN_AGE_DAYS} days')
        AND oo.observation_id IS NULL
        AND (o.charge != 'metabolized' OR o.charge IS NULL)
        AND o.weight IN ('medium', 'heavy')
        AND o.archived_at IS NULL
    `).all();

    for (const orphan of orphans.results || []) {
      await env.DB.prepare(`
        INSERT OR IGNORE INTO orphan_observations (observation_id) VALUES (?)
      `).bind(orphan.id).run();
      orphansIdentified++;
    }

    // 4. Idempotent novelty recalculation
    // novelty = GREATEST(weight_floor, LEAST(1.0, base_decay + time_recovery))
    // Running this 1x or 48x produces the same result.
    await env.DB.prepare(`
      UPDATE observations
      SET novelty_score = GREATEST(
        CASE weight
          WHEN 'heavy' THEN ${NOVELTY_FLOORS.heavy}
          WHEN 'medium' THEN ${NOVELTY_FLOORS.medium}
          ELSE ${NOVELTY_FLOORS.light}
        END,
        LEAST(1.0,
          (1.0 - COALESCE(surface_count, 0) *
            CASE weight
              WHEN 'heavy' THEN ${NOVELTY_DECAY_RATES.heavy}
              WHEN 'medium' THEN ${NOVELTY_DECAY_RATES.medium}
              ELSE ${NOVELTY_DECAY_RATES.light}
            END)
          + CASE
              WHEN last_surfaced_at IS NOT NULL
              THEN LEAST(${NOVELTY_TIME_RECOVERY_CAP},
                EXTRACT(EPOCH FROM (NOW() - last_surfaced_at)) / 86400.0 * ${NOVELTY_TIME_RECOVERY_RATE})
              ELSE 0
            END
        )
      )
      WHERE archived_at IS NULL
        AND (charge != 'metabolized' OR charge IS NULL)
    `).run();

    // 4b. Automatic charge progression
    // fresh -> active: system has engaged with this observation (surfaced 2+ times)
    await env.DB.prepare(`
      UPDATE observations SET charge = 'active'
      WHERE charge = 'fresh'
        AND COALESCE(surface_count, 0) >= 2
        AND archived_at IS NULL
    `).run();

    // active -> processing: deeply familiar or sat with multiple times
    await env.DB.prepare(`
      UPDATE observations SET charge = 'processing'
      WHERE charge = 'active'
        AND (
          COALESCE(surface_count, 0) >= 5
          OR (added_at < datetime('now', '-30 days') AND COALESCE(sit_count, 0) >= 2)
        )
        AND archived_at IS NULL
    `).run();

    // 5. Deep archive pass - fade old observations that were never engaged with
    // Light path: 30 days old, 0 sits, not surfaced in 30d, not foundational
    // Medium path: 90 days old, 0 sits, not surfaced in 60d, not foundational
    const archiveCandidates = await env.DB.prepare(`
      SELECT id FROM (
        SELECT o.id
        FROM observations o
        JOIN entities e ON o.entity_id = e.id
        WHERE o.archived_at IS NULL
          AND o.weight = 'light'
          AND COALESCE(o.sit_count, 0) = 0
          AND (o.last_surfaced_at IS NULL OR o.last_surfaced_at < datetime('now', '-30 days'))
          AND o.added_at < datetime('now', '-${ARCHIVE_AGE_DAYS} days')
          AND (o.charge != 'processing' OR o.charge IS NULL)
          AND COALESCE(e.salience, 'active') != 'foundational'
        UNION
        SELECT o.id
        FROM observations o
        JOIN entities e ON o.entity_id = e.id
        WHERE o.archived_at IS NULL
          AND o.weight = 'medium'
          AND COALESCE(o.sit_count, 0) = 0
          AND (o.last_surfaced_at IS NULL OR o.last_surfaced_at < datetime('now', '-60 days'))
          AND o.added_at < datetime('now', '-90 days')
          AND (o.charge != 'processing' OR o.charge IS NULL)
          AND COALESCE(e.salience, 'active') != 'foundational'
      )
      LIMIT 50
    `).all();

    let archivedCount = 0;
    for (const obs of archiveCandidates.results || []) {
      await env.DB.prepare(`
        UPDATE observations SET archived_at = datetime('now') WHERE id = ?
      `).bind(obs.id).run();
      archivedCount++;
    }
    if (archivedCount > 0) {
      console.log(`Archived ${archivedCount} observations to the deep`);
    }

    // 6. Access-based novelty decay — penalize never-accessed old observations
    try {
      await env.DB.prepare(`
        UPDATE observations
        SET novelty_score = GREATEST(
          CASE weight WHEN 'heavy' THEN 0.2 WHEN 'medium' THEN 0.1 ELSE 0.05 END,
          novelty_score - ${ACCESS_DECAY_PENALTY}
        )
        WHERE archived_at IS NULL
          AND (charge != 'metabolized' OR charge IS NULL)
          AND COALESCE(access_count, 0) = 0
          AND added_at < datetime('now', '-${ACCESS_DECAY_AGE_DAYS} days')
          AND novelty_score > 0.3
      `).run();
    } catch { /* access_count column may not exist yet */ }

    // 7. LLM-driven memory consolidation
    try {
      const consolidatedCount = await consolidateRelatedObservations(env);
      if (consolidatedCount > 0) console.log(`Consolidated ${consolidatedCount} observation groups`);
    } catch (e) {
      console.log(`Consolidation error: ${e}`);
    }

    // 8. Structured reflection from recent observations
    try {
      await generateSessionReflection(env);
    } catch (e) {
      console.log(`Reflection error: ${e}`);
    }

  } catch (e) {
    // Living surface tables might not exist yet - that's fine
    console.log(`Living surface tables not ready: ${e}`);
  }

  // 9. Dream processing — independent, runs during night hours (22:00-05:00)
  try {
    await processDream(env);
  } catch (e) {
    console.log(`Dream processing error: ${e}`);
  }

  // Get counts for orient display
  let pendingProposals = 0;
  let orphanCount = 0;
  let noveltyDist = { high: 0, medium: 0, low: 0 };

  try {
    const proposalCount = await env.DB.prepare(`
      SELECT COUNT(*) as count FROM daemon_proposals WHERE status = 'pending'
    `).first();
    pendingProposals = (proposalCount?.count as number) || 0;

    const orphanCountResult = await env.DB.prepare(`
      SELECT COUNT(*) as count FROM orphan_observations
    `).first();
    orphanCount = (orphanCountResult?.count as number) || 0;

    const noveltyResult = await env.DB.prepare(`
      SELECT
        SUM(CASE WHEN COALESCE(novelty_score, 1.0) > 0.7 THEN 1 ELSE 0 END) as high,
        SUM(CASE WHEN COALESCE(novelty_score, 1.0) BETWEEN 0.4 AND 0.7 THEN 1 ELSE 0 END) as medium,
        SUM(CASE WHEN COALESCE(novelty_score, 1.0) < 0.4 THEN 1 ELSE 0 END) as low
      FROM observations
      WHERE charge != 'metabolized' OR charge IS NULL
    `).first();
    if (noveltyResult) {
      noveltyDist = {
        high: (noveltyResult.high as number) || 0,
        medium: (noveltyResult.medium as number) || 0,
        low: (noveltyResult.low as number) || 0
      };
    }
  } catch {
    // Tables not ready
  }

  // Store state in subconscious table
  const state = {
    processed_at: now.toISOString(),
    hot_entities: hotEntities,
    recurring_patterns: recurring,
    mood: { dominant: dominantEmotion, confidence: totalEmotionSignals >= 10 ? "high" : totalEmotionSignals >= 5 ? "medium" : totalEmotionSignals >= 3 ? "low" : "insufficient" },
    context_clusters: contextClusters,
    // Relation-based analysis
    central_nodes: centralNodes,
    relation_patterns: relationPatterns,
    relation_clusters: relationClusters.slice(0, 5),
    graph_stats: {
      total_relations: allRelations.results?.length || 0,
      unique_relation_types: Object.keys(relationTypeCounts).length,
      connected_entities: Object.keys(connectivity).length
    },
    // NEW: Living surface state
    living_surface: {
      pending_proposals: pendingProposals,
      orphan_count: orphanCount,
      novelty_distribution: noveltyDist,
      strongest_co_surface: strongestCoSurface.slice(0, 3)
    }
  };

  // Upsert into subconscious table
  await env.DB.prepare(`
    INSERT INTO subconscious (id, state_type, data, updated_at)
    VALUES (1, 'daemon', ?, ?)
    ON CONFLICT(id) DO UPDATE SET data = ?, updated_at = ?
  `).bind(JSON.stringify(state), now.toISOString(), JSON.stringify(state), now.toISOString()).run();

  console.log(`Subconscious processed: ${hotEntities.length} hot, ${recurring.length} patterns, ${centralNodes.length} central, ${proposalsCreated} proposals, ${orphansIdentified} orphans`);
}


export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    // Swap D1 + Vectorize for Postgres adapters via Hyperdrive
    const pgEnv = env.HYPERDRIVE
      ? {
          ...env,
          DB: createD1Adapter(env.HYPERDRIVE) as unknown as D1Database,
          VECTORS: createVectorAdapter(env.HYPERDRIVE.connectionString) as unknown as VectorizeIndex,
        }
      : env;
    return routeRequest(request, pgEnv, {
      processSubconscious,
      handleApiEntities,
      handleApiObservations,
      handleApiJournals,
      handleApiThreads,
      handleApiSearch,
      handleApiSurface,
      handleApiIdentity,
      handleApiRelations,
      handleApiImages,
      handleApiContext,
      handleApiBulkObservations,
      handleApiProcess,
      handleApiOrient,
      handleApiGround,
      handleApiHealth,
      handleApiHealthScores,
      handleApiStats,
      handleApiHeat,
      handleApiRecent,
      handleApiInnerWeather,
      handleApiPatterns,
      handleApiTensions,
      handleApiProposals,
      handleApiOrphans,
      handleApiArchive,
      handleApiObservationVersions,
      handleMCPRequest
    });
  },

  async scheduled(event: ScheduledEvent, env: Env, ctx: ExecutionContext): Promise<void> {
    const pgEnv = env.HYPERDRIVE
      ? {
          ...env,
          DB: createD1Adapter(env.HYPERDRIVE) as unknown as D1Database,
          VECTORS: createVectorAdapter(env.HYPERDRIVE.connectionString) as unknown as VectorizeIndex,
        }
      : env;
    ctx.waitUntil(processSubconscious(pgEnv));
  }
};
