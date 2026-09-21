// Constraint 8 of the Interior Build Plan (21 Sep 2026): metabolized is never automatic.
// Fails if any statement inside the daemon (processSubconscious*) writes charge='metabolized'.
// Reads in WHERE clauses are fine; SET/INSERT with that value are not.
import { readFileSync } from "node:fs";
const src = readFileSync(new URL("../src/index.ts", import.meta.url), "utf8");
const start = src.indexOf("async function processSubconscious(");
const end = src.indexOf("\nexport default {", start);
if (start < 0 || end < 0) { console.error("guard: could not locate the daemon"); process.exit(2); }
const daemon = src.slice(start, end);
const re = /'metabolized'/g;
let m, bad = [], seen = 0;
while ((m = re.exec(daemon))) {
  seen++;
  const back = daemon.slice(Math.max(0, m.index - 160), m.index);
  const kws = [...back.matchAll(/\b(SET|WHERE|AND|OR|VALUES|INSERT|WHEN|THEN)\b|(!=|<>|===|!==)/g)];
  const last = kws.length ? kws[kws.length - 1][0] : "";
  if (last === "SET" || last === "VALUES" || last === "INSERT") {
    const line = src.slice(0, start + m.index).split("\n").length;
    bad.push(`src/index.ts:${line}: ...${back.slice(-80).replace(/\s+/g, " ")}'metabolized'`);
  }
}
if (bad.length) { console.error("guard FAILED — the daemon writes charge='metabolized':\n" + bad.join("\n")); process.exit(1); }
console.log(`guard OK — no daemon path writes charge='metabolized' (${seen} read-only mentions checked)`);
