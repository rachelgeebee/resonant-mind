-- The light register (Interior Build Plan 2.1, agreed with Rachel 3 Oct 2026).
-- The fridge-door note: one line, kept at the size it happened. Two kinds only.
--   small_joy     something that delighted, remembered small
--   quietly_want  a want that needs no action; sits until a moment meets it, then met
-- Not an observation (no entity/weight/charge, never in surfacing pools or graph stats).
-- Not a drive (nothing counts it, nothing escalates it). The daemon only lets rows fade.
-- `who` is whose line it is; `author` is which instance/channel typed it.
-- Timestamps are ISO-8601 UTC (JS toISOString).
CREATE TABLE IF NOT EXISTS register (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL CHECK (kind IN ('small_joy', 'quietly_want')),
    who TEXT NOT NULL DEFAULT 'theo' CHECK (who IN ('rachel', 'theo', 'us')),
    author TEXT,
    text TEXT NOT NULL CHECK (length(text) <= 140),
    created_at TEXT NOT NULL,
    met_at TEXT,
    met_note TEXT,
    faded_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_register_created ON register(created_at);
