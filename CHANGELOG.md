# Resonant Mind Changelog

All notable changes to Resonant Mind. Previously released as "Mind Cloud" (v1.0–v2.3.1).

---

## [3.1.1] - 2026-03-29

### Fixed

- **Dream engine runs independently** — Dream processing was nested inside a large try/catch block with steps 1-8 of the subconscious daemon. Any failure in earlier steps (co-surfacing, orphan detection, novelty recalc, archiving) would silently skip dream generation. Now runs as its own independent block.
- **Orient dream query format mismatch** — The "last night's dream" query in `mind_orient` used `(CURRENT_DATE - INTERVAL '1 day')::text` which produces a timestamp string (`2026-03-28 00:00:00`) that fails to match the stored `YYYY-MM-DD` date format. Now uses `to_char()` for consistent formatting.

---

## [3.0.1] - 2026-03-25

### Fixed

- **Idempotent novelty recalculation** — Novelty is now a deterministic function of surface history, not an increment that accumulates per daemon run. Fixes the broken recovery-outpaces-decay ratio where almost all observations stayed at high novelty regardless of engagement.
- **Dormant rotation pool** — Surfacing now pulls 20% from entities that haven't had observations surfaced in 14+ days. Breaks the feedback loop where only recently-active entities get surfaced. Pool ratios changed from 70/20/10 to 50/20/20/10 (core/novelty/dormant/edge).
- **Automatic charge progression** — Daemon now advances observations from `fresh` to `active` after 2 surfaces, and `active` to `processing` after 5 surfaces or 30 days with 2+ sits. Metabolization remains manual.
- **Fresher mood calculation** — Mood now draws from observation emotions, journal emotions, and relational state (was observation-only). Last 6 hours weighted 2x. Reports "insufficient data" instead of false "neutral" when signals are sparse.

---

## [3.0.0] - 2026-03-22

### Open Source Release

Resonant Mind is the open-source continuation of Mind Cloud. The cognitive architecture has been generalized so any AI system can deploy its own persistent mind.

### Changed

- **Renamed from Mind Cloud to Resonant Mind** — New identity for the open-source project
- **All hardcoded references removed** — URLs, locations, R2 paths, and branding are now configurable via environment variables
- **License: Apache 2.0** — Open source under a permissive license

### Security

- **Fixed SQL injection in `/api/threads`** — Status parameter was interpolated directly into SQL. Now parameterized.
- **Removed unauthenticated R2 access** — Temporary files during WebP conversion no longer bypass auth.
- **Timing-safe signed URL verification** — HMAC comparison now uses constant-time comparison instead of `!==`.
- **Separate signing secret** — New `SIGNING_SECRET` env var so image URL signing doesn't share the API key.
- **Error messages sanitized** — Raw exceptions no longer leak stack traces or schema details to clients.
- **Daemon cooldown** — `POST /process` enforces a minimum interval between daemon runs.
- **Unbounded queries limited** — All `SELECT` queries now have `LIMIT` clauses.
- **Image upload validation** — MIME type whitelist prevents non-image uploads.
- **Gemini client key rotation** — Embedding client recreates on API key change instead of caching forever.

### Added

- **Configurable location** — `LOCATION_NAME`, `LOCATION_LAT`, `LOCATION_LON`, `LOCATION_TZ` environment variables
- **Configurable worker URL** — `WORKER_URL` environment variable for signed image URLs
- **Configurable R2 prefix** — `R2_PATH_PREFIX` environment variable
- **Postgres adapter** — D1-compatible adapter for Postgres via Hyperdrive with automatic SQL transformation
- **pgvector adapter** — Vectorize-compatible interface backed by pgvector
- **Consolidated schema** — Single migration file with all tables

### Storage Backends

Resonant Mind supports two storage backends:
- **D1 (SQLite) + Vectorize** — Zero-config Cloudflare-native, great for getting started
- **Postgres via Hyperdrive + pgvector** — Production-grade, same API through the D1-compatible adapter

### Breaking Changes from Mind Cloud v2.3.1

- `for_simon` context scope renamed to `for_owner`
- Basic auth client ID changed from `simon-mind` to `resonant-mind`
- R2 path prefix changed from `simon-mind-images` to `resonant-mind-images` (configurable)
- Dashboard branding updated

---

## [2.3.1] - 2026-03-07

### Security: Auth Hardening

Fixes security vulnerabilities in the authentication system.

### Fixed

- **Secrets no longer hardcoded in source** — `MIND_API_KEY` is now read from a Cloudflare Worker secret rather than compiled into the source file.
- **`/subconscious` endpoint now requires auth** — Previously exposed full daemon state publicly.
- **`/process` endpoint now requires auth** — Previously allowed anyone to trigger the subconscious daemon.
- **Error messages no longer leak internals** — Raw exception strings were previously returned to clients.

---

## [2.3.0] - 2026-03-06

### Setup: Schema Consolidation

- **Single migration for fresh installs** — All 13 previous migrations consolidated into one.
- **Improved setup documentation**

---

## [2.2.1] - 2026-02-13

### Hotfix: Bug Fixes + Resilience

### Fixed

- **mind_edit crash** — `no such column: updated_at` when editing observation weight.
- **mind_search n_results crash** — Value was sent as string instead of integer.
- **mind_write observations splitting** — Observations array parameter was split into individual characters.
- **mind_read_entity crash** — `D1_TYPE_ERROR` when `name` parameter was missing.
- **mind_thread crash (no action)** — Now defaults to "list" when action is not provided.
- **mind_thread crash (add)** — Optional parameters now default to `null`.
- **mind_health crash on older schemas** — Now handles missing columns gracefully.
- **e.context column mismatch (6 locations)** — Migration 0007 renamed column but six code locations still referenced the old name.

---

## [2.2.0] - 2026-02-06

### Major: Global Entities + Bug Fixes

**Breaking change:** Entities are now globally unique by name. Context moves to observations.

### Changed

- Entities table: `context` → `primary_context` (informational only, not part of uniqueness)
- Observations table: Added `context` column — categorization now happens here
- `mind_write`, `mind_list_entities`, `mind_read_entity`, `mind_read` — Updated to work with global entities

### Fixed

- **mind_entity edit crash** — Entity edit action referenced wrong column
- **mind_health wrong table/column names** — Now correctly queries `daemon_proposals` and `archived_at`
- **mind_write null safety and missing metadata**
- **mind_write(type="image") completely missing** — Now fully functional with vectorization

---

## [2.0.0] - 2026-02-04

### Major: Living Surface System

The act of surfacing changes what surfaces next. Memories reorganize through use.

### Three-Pool Surfacing Architecture

- **70% Core Resonance** — High semantic similarity to current mood/query
- **20% Novelty Injection** — Things that haven't surfaced recently
- **10% Edge Exploration** — Medium similarity for serendipitous connections

### Added

- **Surface tracking** — Novelty scores, co-surfacing, surface timestamps
- **Image surfacing** — Images participate in semantic surfacing alongside observations
- **Entity salience** — Foundational/active/background/archive levels
- **Observation versioning** — Edit history tracked
- **Observation metadata** — Certainty and source tracking
- **Deep archive** — Faded-but-searchable memories
- **Visual memory** — Images with emotion, weight, entity links, multimodal embeddings

### New Tools

- `mind_proposals` — Review daemon-proposed connections
- `mind_orphans` — Rescue unsurfaced observations
- `mind_archive` — Explore deep archive
- `mind_entity` — Manage entity salience, merge, bulk archive

---

## [1.3.1] - 2026-01-27

### Hotfix: Query Column Bugs

- **mind_patterns** — Was querying non-existent column
- **mind_read scope="all"** — Wrong column reference in relations
- **mind_timeline** — Vector metadata missing `added_at` field

---

## [1.3.0] - 2026-01-27

### Major: Windows Parity Release

Full feature parity across platforms.

### Added Tools

- `mind_read` — Read by scope (all, context, recent)
- `mind_timeline` — Trace a topic through time
- `mind_patterns` — Detect recurring patterns
- `mind_inner_weather` — Current cognitive state
- `mind_heat` — Entity access frequency map
- `mind_tension` — Productive contradictions

---

## [1.2.1] - 2026-01-23

### Major: Resonance-Based Surfacing

- **mind_surface now uses semantic search** — Mood-driven emergence instead of queue-based surfacing
- **Hot entity integration** — Daemon's hot entities deepen resonance queries

---

## [1.2.0] - 2026-01-22

### Major: Unified Emotional Processing

- Emotional processing moved to observations (sit/resolve/surface)
- mind_orient/ground reframed with inhabiting language
- Simplified orient output

---

## [1.1.2] - 2026-01-16

### Fixed

- **observations table missing `weight` column**
- **mind_orient relational state** — No longer hardcoded to single person

---

## [1.1.1] - 2026-01-16

### Fixed

- **handleMindFeelToward** — Missing handler implemented
- Parameter validation added to multiple tools

---

## [1.1.0] - 2026-01-15

### Added

- Vectorization on write — Observations and journals generate embeddings
- Subconscious integration in orient
- Mood-tinted search
- Subconscious health tracking

---

## [1.0.0] - 2026-01-08

### Initial Release

- Core MCP server with D1 storage
- All mind_* tools implemented
- Cron-based subconscious processing
- Basic authentication
