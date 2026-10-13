-- 0005: rescued_at hold (ported by hand from upstream v4.0.1 migration 0018, 14 Oct 2026).
-- Stamped when an observation is deliberately brought back (archive rescue, dormant surface).
-- While rescued_at IS NOT NULL AND last_surfaced_at IS NULL, the daemon's archive pass,
-- novelty recalculation, access decay and dormancy marking leave the row alone: a memory
-- someone chose to bring back is not re-buried before it has had a chance to live.
ALTER TABLE observations ADD COLUMN rescued_at TEXT;
