# Fly Doctrines

A fruit fly's brain plays one country in [OpenDoctrines](https://github.com/Pr1nted/Open-Doctrines).

Not a heuristic named after a fly. The whole [FlyWire](https://flywire.ai)
connectome — **138,639 neurons, 15,091,983 synapses** — simulated as leaky
integrate-and-fire, the model from Shiu et al., *Nature* (2024), wired to the
game's own decision points. ParrotZero, the game's AI, keeps playing everyone
else.

## How a war reaches a fly

A fly has four senses, so the position is reduced to four firing rates — the
same encoding [Open Fly](https://github.com/Pr1nted/Open-Fly) uses:

| Channel | What it carries |
|---|---|
| `sugar` | land gained since last turn, and profit |
| `bitter` | land lost, losses, and wars newly declared |
| `water` | treasury, measured in turns of reserve |
| `jon` | how many wars it is in (Johnston's organ — hearing) |

200 ms of brain time later, the **1,299 descending neurons** — the ones that in
a real fly carry commands from brain to body — have each spiked some number of
times. They are dealt into as many groups as the decision has legal moves, and
the group that fired hardest picks the move.

Except that "hardest" cannot mean the raw count, and finding out why was the
most interesting part of building this. **Descending neurons are not equally
excitable**: a handful fire hard in every window whatever the fly saw, so
whichever group they land in wins every turn. Measured over fifteen positions
with the senses swinging from 0 to 200 Hz, the answer was action 0 on all
fifteen — a mod that looked like it worked and was quietly ignoring the game.

So the decoder reads how much each neuron fired **relative to its own recent
baseline**. A neuron that always fires says nothing by firing again; one firing
more than it usually does is the signal. That is also roughly how sensory
systems behave — adaptation, not absolute rate. The first window of a game
therefore establishes the baseline and decides nothing; ParrotZero plays that
turn.

Nobody knows which fly neuron should mean "raise taxes", and pretending to would
be the dishonest part of this. The deal is seeded and deterministic, so the same
fly plays the same way on the same position — but it is a deal, not a claim.

**It does not play well.** It is a real neural simulation attached to a strategy
game, and it plays about as well as that sounds. The point is that every spike
is the model's.

## The three numbers it was built around

All measured, none assumed.

**The connectome is 91 MB and a mod gets 64 MiB.** The first idea was to prune
to the neurons on a path from a sense to a descending neuron. Measured: that
keeps **97.7% of neurons and 99.9% of synapses** — the fly brain is densely
connected and there is nothing to cut. What worked instead is delta+varint
encoding of the sorted CSR, which is **lossless** and gets it to **42.8 MB**. No
neuron dropped, no synapse dropped, no timestep coarsened. `tools/`
`pack_connectome.py --verify` decodes all 15,091,983 edges back and compares.

**One decision is ~19.1M neuron integrations**, past the 100M per-turn fuel
ceiling on its own, which is why the manifest declares `fuelPerDecision`. The
simulation is event-driven: of 138,639 neurons a 200 ms window moves a few
thousand off rest, and the rest cost nothing.

**And the fuel cost depends on the position**, which was the last thing to go
wrong and the hardest to have guessed. Counting operations by hand suggested
~154M instructions. Through WAMR a *quiet* turn measures 700M–1,000M — about six
times that, because an interpreter's instruction count is not an operation
count. A *busy* turn, with a war on and all four senses firing, exceeded
**2,000,000,000** and trapped.

That is past any budget a mod can be granted: the runtime's instruction limit is
an `int`, so `INT32_MAX` is the wall. And a trap is not a lost decision — the
host disables the mod. So the fly gets a **bounded amount of thinking**, counted
in integrations plus synapses decoded. A quiet turn never reaches the bound and
runs the whole 200 ms; a busy one thinks for less of it, which is a fair
description of a small brain under load.

**One decision costs ~3 s** under WAMR (0.8 s in Node — the interpreter is the
difference), which is why it plays exactly one country and runs the brain
**once per turn** rather than once per decision module. Deciding for every AI
country would put minutes on every turn.

## Is it really the same brain?

`tools/validate.c` runs a window in C and `tools/reference.mjs` runs the same
window in Open Fly's JavaScript, and both print a fingerprint over every
spiking neuron and its count:

```
  spikes 4976  neurons that fired 601  descending spikes 336
  fingerprint 679c4d939d4ab64b
```

Identical. Same spikes, same neurons, same counts, same places — not "close".
A port of a spiking simulator that is subtly wrong still produces plausible
spikes, so a total alone would not have been evidence.

`tools/decide_test.c` then stubs the host and drives real decisions, because a
mod that only *loads* has none of its deciding exercised. It pins ten things,
including the two that were wrong at first: that a changing position changes the
answer, and that the first window defers.

Finally, under the real runtime:

```bash
odmod-check fly-doctrines.odmod --decide 24
```

which instantiates the actual wasm in WAMR with the real host imports and the
real fuel budget, and drives 24 turns of a moving world. That is the step that
caught the position-dependent fuel cost — nothing before it was running the
module under metering.

## Building it

Needs a wasm32-capable clang and an OpenDoctrines checkout for `sdk/gearbox.h`.

```bash
python3 tools/pack_connectome.py --fly ../Open-Fly --out data/brain.odfly --verify
OD=../OpenDoctrines ./build.sh
```

Then check the host will have it:

```bash
../OpenDoctrines/build/odmod-check fly-doctrines.odmod
```

## Requires Gearbox 1.3

It needs `Neural.Decide`, the capability that lets a mod choose the AI's move
rather than only watch it. `Neural` on its own is read-only, and the split is
deliberate: a player who agreed to *read what the AI sees* has not agreed to
*choose what it does*.

`side` is `server`, so only the host needs it — the AI is decided
authoritatively and joiners need nothing.

## Credit

The connectome and the model are not mine. FlyWire, and Shiu et al. 2024, *A
leaky integrate-and-fire computational model based on the connectome of the
entire adult Drosophila brain reveals insights into sensorimotor processing*.
The packing, the port and the very poor strategic judgement are.

## Licence

**CC BY-NC 4.0** — <https://creativecommons.org/licenses/by-nc/4.0/>

The whole mod, not just the data. 42 of the 42.8 MB in `fly-doctrines.odmod` is
the FlyWire connectome, which FlyWire publishes under CC BY-NC 4.0, so the
package takes the terms of the thing it is mostly made of. It also matches the
game, which is non-commercial itself.

Share it, change it, run it. Do not sell it or put it inside something sold.
Attribution is required and `LICENSE` says exactly what to attribute.
