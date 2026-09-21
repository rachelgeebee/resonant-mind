-- Daemon run log (Interior Build Plan 1.3, 21 Sep 2026).
-- One row per processSubconscious() run: when, whether it finished clean, what moved.
-- `stats` is JSON with one count per step (archived, charged_active, charged_processing,
-- proposals, dormant_marked, dormant_cleaned, novelty_updated, access_decayed, context_expired).
-- Timestamps are ISO-8601 UTC (JS toISOString), consistently, so range queries can bind ISO cutoffs.
CREATE TABLE IF NOT EXISTS daemon_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    ok INTEGER NOT NULL DEFAULT 1,
    error TEXT,
    stats TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_daemon_runs_started ON daemon_runs(started_at);
