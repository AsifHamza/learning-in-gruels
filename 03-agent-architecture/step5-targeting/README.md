# Step 5 — Who does it shoot?

**The diagnostic that survival numbers cannot replace.**

> **This folder stands alone.** It has its own copy of `gruel_env.py`, its own
> `requirements.txt`, and imports nothing from any other folder. Copy it anywhere and it runs.

Player count here: **N=3** for everything the exact answer covers — plus a short **N=5**
block for one instrument, because N=3 cannot exercise it. The reason is given below and it
is a property of the game, not a convenience.

## Why this step exists

**Why survival numbers are not enough.** A flat survival profile — everyone surviving about
1/N of the time — is produced by **two completely different policies**: an equilibrium that happens to be
near-uniform, and a policy that has learned nothing and discriminates on nothing. The survival numbers are
identical. No amount of extra episodes separates them.

**What separates them is who the agent shoots.** Three signatures, all distinguishable:

| Policy | P(shoot the stronger neighbour) |
|---|---|
| matched the equilibrium | ≈1.0 at the extremes, ≈0.3 in the middle, and it **changes as players die** |
| imitating `shoot_strongest` | high everywhere, flat |
| discriminating on nothing | ≈0.5 everywhere, and it does **not** change as players die |

**Why it must be conditioned on how many are alive.** The equilibrium's own targeting is not one number:
at five alive it fires at the stronger neighbour 40% of the time, at four alive 50%, at three alive 100%.
Pooling those into a single average describes a game nobody plays.

**And why that dimension is empty at N=3.** Play begins with three alive; the first kill leaves two; and
a two-player state is excluded, because a player whose two neighbours are the *same person* has made no
choice between them. So every countable decision at N=3 has exactly three alive — **one row**. A single
row showing no variation reads exactly like "discriminates on nothing", and is nothing of the kind: there
is no second row to differ from. Since step 6 hands down the first pass or fail and reads this
instrument, it is calibrated here at N=5, against a policy built to have a known survivor dependence.

**Why calibrate at N=3.** Because there the exact answer is known before the instrument runs: the
equilibrium at N=3 *is* `shoot_strongest`, so every player should fire at its stronger neighbour with
probability 1.0. An instrument that cannot reproduce a known answer cannot be trusted on an unknown
one.

## What it contains

- Recording, for each shot, who fired and which neighbour was stronger — captured
  **before** the step, because a kill contracts the circle and destroys the context.
- The three rules that keep the measurement honest: the shooter must have **fired**, its two
  neighbours must be **distinct**, and they must have **different** accuracies. The first is why a
  policy that never fires reports *no data* instead of a rate; the second excludes every
  two-player state, where both neighbours are the same person.
- Targeting by accuracy rank, and separately by how many players remain.
- The three signatures above, measured on heuristics and then on the trained agent.

## What number to expect

**The instruments, against answers known in advance.** `shoot_strongest` measures exactly `1.0000`
and `shoot_weakest` exactly `0.0000`; `coin_flip` lands at `0.4886`; `always_miss` reports **no data
from 0 decisions**, not a rate of zero.

**The policy built to be misdescribed by an average.** `split_by_seat` measures `1.0000` / `0.0000` /
`1.0000` by rank and pools to `0.6661` — a figure describing no player in it.

**The survivor-count instrument, at N=5.** The constructed step is recovered exactly: `switch_at_three`
gives `0.0000` at five alive, `0.0000` at four, `1.0000` at three. The two flat policies stay flat —
`shoot_strongest` at `1.0000` throughout, `coin_flip` at `0.5001` / `0.5137` / `0.4956`.

**The trained agent** (38,400 episodes of self-play, 6,000 measured, stagnation `0.0000`):

| accuracy | P(shoot stronger) | survival | ratio to exact |
|---|---|---|---|
| 0.4 weakest | `0.9559` | `0.3598` | 0.972 |
| 0.6 middle | `0.9959` | `0.3350` | 1.005 |
| 0.8 strongest | `0.9994` | `0.3052` | 1.030 |

Pooled targeting `0.9839`, spread across ranks `0.0435`. **Flat and near 1.0 — and at N=3 that is
correct**, because the equilibrium there *is* `shoot_strongest`. The measurement therefore cannot
separate "matched the equilibrium" from "imitated a heuristic": the two references tie at a distance of
`0.0132` because at N=3 they are the same policy. Step 6 is the first player count at which they differ.

**Survival and targeting are independent axes**, checked rather than asserted: sorting the policies by
survival spread gives a different order from sorting them by targeting. `shoot_weakest` is the most
differentiated policy by survival and the lowest by targeting.

## What to watch in the output

**That the N=3 result is flat, and that this is correct.** Looking for a U-shape here would be
looking for the wrong thing — the shape only emerges once the graph constraint starts to bind.

**That two references tie.** The report says so rather than picking one. Naming a single "closest"
reference where two are the same policy would invent a distinction that the measurement did not make.

**That `always_miss` reports no data.** A rate of `0.0` would be the specific claim *"it always shot
the weaker neighbour"* — a statement about behaviour that never happened.

## Reproducibility

The notebook ends by running the same thing twice from the same seed and printing whether the
two results are identical. A step that cannot reproduce itself cannot support a claim.

## Where this sits

- Previous: [`../step4-measurement/`](../step4-measurement/) — Build the instruments, and check them against answers already known.
- Next: [`../step6-n5-gate/`](../step6-n5-gate/) — The first real pass or fail, against a profile that is not monotone.
- The whole investigation: [`../README.md`](../README.md)
