// The same window, from Open-Fly's own JavaScript brain, as the reference the
// C port must match.
//
//   node tools/reference.mjs <Open-Fly checkout>
//
// Prints the same three numbers and the same fingerprint as tools/validate.c,
// computed the same way, so the two can be compared without either side
// knowing what the other expects.
import { readFileSync } from "node:fs";

const fly = process.argv[2];
if (!fly) { console.error("usage: node tools/reference.mjs <Open-Fly checkout>"); process.exit(2); }

const { FlyBrain } = await import(`${fly}/web/brain.js`);
const meta = JSON.parse(readFileSync(`${fly}/web/data/brain.json`, "utf8"));
const c = readFileSync(`${fly}/web/data/connectome.bin`);
const brain = new FlyBrain(c.buffer.slice(c.byteOffset, c.byteOffset + c.byteLength), meta);

// Must match tools/validate.c exactly, bitter included: a zero channel draws no
// random numbers, and getting that wrong shifts every later draw.
const rates = { sugar: 50, bitter: 0, water: 10, jon: 100 };
const counts = brain.runWindow(rates, 200.0, 20240922 >>> 0);

let total = 0, active = 0, dnTotal = 0;
let hash = 0xcbf29ce484222325n;
const FNV = 0x100000001b3n, MASK = (1n << 64n) - 1n;
for (let i = 0; i < counts.length; i++) {
  if (!counts[i]) continue;
  total += counts[i]; active++;
  const b = [i & 255, (i >>> 8) & 255, (i >>> 16) & 255, (i >>> 24) & 255,
             counts[i] & 255, (counts[i] >>> 8) & 255];
  for (const x of b) { hash = ((hash ^ BigInt(x)) * FNV) & MASK; }
}
for (const i of meta.dn) dnTotal += counts[i];

console.log(`  spikes ${total}  neurons that fired ${active}  descending spikes ${dnTotal}`);
console.log(`  fingerprint ${hash.toString(16).padStart(16, "0")}`);
