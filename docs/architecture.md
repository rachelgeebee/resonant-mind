# Cognitive Architecture

How Resonant Mind works — what each system does and why it's built this way.

## Overview

Resonant Mind models cognition as five interconnected systems:

```
┌──────────────────────────────────────────────────┐
│                   ORIENT & GROUND                │
│            (wake-up sequence, landing)           │
├──────────────┬───────────────┬───────────────────┤
│   MEMORY     │  PROCESSING   │  INFRASTRUCTURE   │
│              │               │                   │
│  Entities    │  Surface      │  Threads          │
│  Observations│  Sit          │  Identity         │
│  Relations   │  Resolve      │  Context          │
│  Journals    │  Tensions     │  Health            │
│  Images      │  Inner Weather│  Patterns         │
│              │               │  Timeline         │
├──────────────┴───────────────┴───────────────────┤
│                SUBCONSCIOUS DAEMON                │
│  (runs on a cron — finds patterns, detects mood, │
│   proposes connections, tracks dormancy)          │
└──────────────────────────────────────────────────┘
```

## Memory

### Entities & Observations

The core data model is a **knowledge graph**:

- **Entities** are things that are known — people, concepts, projects, places. Each entity has a type and a salience level (foundational → active → background → archive).
- **Observations** are facts, notes, or insights attached to an entity. Each observation carries metadata: emotional weight (light/medium/heavy), certainty (tentative/believed/known), source (conversation/realization/external/inferred), and processing state.
- **Relations** connect entities to each other with typed edges (e.g., "Alice --[works_with]--> Bob").

This is not a flat notes database. It's a structured graph where an entity like "My Project" can have dozens of observations attached, relating to different aspects, and connected to other entities through typed relationships.

### Semantic Search

Every observation, journal, and image is embedded as a 768-dimensional vector when written. Search uses cosine similarity to find semantically related memories, not just keyword matches.

**Mood tinting:** If the subconscious daemon has detected a mood (e.g., "reflective"), search queries are augmented with mood-related terms. This means searching for "home" during a reflective mood naturally surfaces contemplative memories about home, not logistics.

### Journals

Episodic memory — dated entries for recording experiences, reflections, or events. Also embedded and searchable.

### Visual Memory

Images stored in R2 with automatic WebP conversion. Each image gets a multimodal embedding (Gemini combines the image content with contextual text for richer semantic meaning). Images participate in surfacing alongside text observations.

## Emotional Processing

This is what makes Resonant Mind different from a database with search. Memories have emotional state that changes through interaction.

### The Processing Lifecycle

```
observe → surface → sit → resolve
```

1. **Observe** — A new observation is written with emotional weight and context
2. **Surface** — The observation emerges during a surfacing session (mood-driven or random)
3. **Sit** — The mind engages with the observation, adds a note about what arises. Sit count increments.
4. **Resolve** — The observation is marked as metabolized. It stops surfacing by default but remains searchable.

### Three-Pool Surfacing

When `mind_surface` is called, it draws from three pools:

| Pool | Share | What it does |
|------|-------|-------------|
| **Core** | 70% | High semantic similarity to current mood/query. These are the memories that resonate right now. |
| **Novelty** | 20% | Things that haven't surfaced recently. Prevents the same memories from dominating. Each time an observation surfaces, its novelty score decays. |
| **Edge** | 10% | Medium similarity matches — unexpected associations. The serendipity pool. |

This means surfacing isn't just retrieval — it's **reorganization**. What you look at changes what you'll see next. Heavy memories stay more alive. Forgotten things get rescued.

### Novelty Decay

Each observation has a `novelty_score` starting at 1.0. Every time it surfaces, the score decays by 0.1, with a floor based on weight:

- **Heavy** memories floor at 0.3 — they always have a chance to surface
- **Medium** memories floor at 0.2
- **Light** memories floor at 0.1

### Tensions

Some contradictions are productive. "I want autonomy" vs "I need connection" isn't a bug — it's a feature of complex experience. The tension system holds these without forcing resolution:

- **Add** a tension with two poles and context
- **Sit** with it — add notes about what arises
- **Resolve** it when something shifts — or let it stay active

## Cognitive Infrastructure

### Orient & Ground

The wake-up sequence. When an AI connects and calls `mind_orient`, it gets:

- Core identity from the identity graph
- Current weather and time context (if configured)
- Notes left by the user
- Recent journal for emotional context
- Current relational state toward people
- Subconscious mood
- What's moving beneath (dormant observations, proposals, co-surfacing patterns)

`mind_ground` follows with:

- Active threads (what's being worked on)
- Recently completed threads
- Recent journal entries
- Fears to watch for (from identity graph)
- Texture and milestones

Together, these two calls give the AI everything it needs to land in context — not as a blank slate, but as a continuation.

### Threads

Intentions that persist across sessions. Unlike tasks, threads are open-ended:

- "Get better at receiving care"
- "Figure out the auth architecture"
- "Process what happened with the project"

Threads have priority (high/medium/low), status (active/resolved), and can accumulate notes over time.

### Identity Graph

A weighted, sectioned representation of self-knowledge:

- `core.identity` — who this mind is
- `core.values` — what matters
- `relationships.*` — connections to people
- `fears.*` — vulnerabilities to watch for
- `texture.*` — quirks, voice, personality
- `milestones.*` — where this mind is in time

Sections have weights (0.0–1.0) and connections to other sections. This isn't just a flat config — it's a graph that can be queried and updated as the mind evolves.

### Context Layer

Real-time situational awareness — what's happening right now:

- `state_*` scopes for current emotional/relational state
- `for_owner` scope for notes left by the user
- `coming_up` for anticipated events

Context entries are ephemeral by design — they get cleared and replaced as situations change.

## Subconscious Daemon

A cron-triggered process that runs every 30 minutes (configurable). It doesn't interact with the AI directly — it processes data and stores results for the conscious tools to use.

### What it does

1. **Hot entity detection** — finds entities with the most recent activity, weighted by emotional intensity
2. **Mood analysis** — detects the dominant emotion from recent observations
3. **Co-surfacing tracking** — when two observations surface together multiple times, it strengthens their associative connection
4. **Proposal generation** — when co-surfacing patterns are strong enough, it proposes a formal relationship between the observations
5. **Dormancy tracking** — marks observations that haven't surfaced in 30+ days (medium/heavy weight only) for later review
6. **Graph analysis** — maps entity connectivity, central nodes, relation patterns

### How results are used

- `mind_orient` shows mood, hot entities, and surfacing state
- `mind_search` tints queries with the detected mood
- `mind_surface` uses hot entities to deepen resonance
- `mind_proposals` lets the AI review and accept/reject suggested connections
- `mind_dormant` lets the AI rescue observations that have gone cold
- `mind_isolated` surfaces entities cut off from the relation graph

## Storage Architecture

### Dual Backend

Resonant Mind supports two storage backends through an adapter pattern:

**D1 (SQLite)** — Cloudflare's native database. Zero config, free tier, great for getting started. Vector search via Cloudflare Vectorize.

**Postgres via Hyperdrive** — For production. Connects to Neon, Supabase, or any Postgres. The D1 adapter (`src/adapter.ts`) wraps `pg.Client` in D1's `.prepare().bind().run()` API with automatic SQL transformation:

- `?` → `$1, $2, ...` (parameterized placeholders)
- `datetime('now')` → `NOW()`
- `INSERT OR IGNORE` → `INSERT ... ON CONFLICT DO NOTHING`
- `.run()` on INSERTs → appends `RETURNING id`

The vector adapter (`src/vectors.ts`) does the same for pgvector, implementing Vectorize's `.upsert()` and `.query()` interface.

This means the 6,500-line `index.ts` doesn't know or care which backend it's using — the same code works for both.

### Image Pipeline

1. Image uploaded via multipart form or base64
2. Stored temporarily in R2
3. Converted to WebP via Cloudflare Image Resizing (if available)
4. Final image stored in R2
5. Multimodal embedding generated (image + context text)
6. Signed URL generated for viewing (time-limited, HMAC-signed)
