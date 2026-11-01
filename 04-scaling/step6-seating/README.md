# Step 6 — Seating, at 10 and 15 players

Deliverable 03's step 7 changed only the seating at five players and found that it changes which policy
is learned. Under `ranked` seating the agent found the equilibrium; under `random` seating it played
close to `shoot_strongest`. Step 3 of this folder then found a deep U-shaped targeting profile at 10 and
15 players under ranked seating. This step asks whether random seating still pushes the agent towards
`shoot_strongest` once the field is larger.

One dial moves: `ranked` against `random`. The reward stays `terminal`, and every other setting matches
step 1.

## What number to expect

`terminal` reward, 2000 updates × 96 episodes per run, seeds 0 to 2, 10,000 measured episodes per seed.
Twelve training runs. About 78 minutes with four notebooks running at once on a laptop CPU; wall-clock
time varies.

**Survival of the weakest, as a multiple of an even 1/N share:**

| N | ranked | random | difference | combined SE |
|---|---|---|---|---|
| 10 | 1.503 | 1.567 | +0.063 | 0.468 |
| 15 | 1.516 | 2.227 | +0.710 | 0.408 |

Neither difference clears twice its standard error. At N=15 the random arm's weakest player does better
on average, but the three seeds spread from 1.704 to 2.940, so it is not a measurable difference. The
spread under random seating is wide at both counts.

**Targeting, P(shoot the stronger neighbour), is where seating acts.** The two arms learned different
policies at both counts, and by a wide margin: mean distance 0.4507 at N=10 and 0.5402 at N=15, against
the 0.10 threshold.

The difference is in shape. The ranked arm keeps the deep U of `next_weaker`, the fixed rule that always shoots left under ranked seating: the two ends shoot the
stronger neighbour almost every time and the middle rarely does. The random arm is nearly **flat**,
shooting the stronger neighbour about three times in four at every rank, from 0.6280 to 0.7530 at N=10
and 0.7321 to 0.7595 at N=15.

**At these sizes random seating does not reproduce `shoot_strongest`, which it did at five players.**
Measured against three references in the same game, the ranked arm matches `next_weaker`, 0.0424 away at N=10 and 0.0225 at N=15, as step 3 found. The random arm matches none: it is nearer `shoot_strongest` than the ranked arm is (0.3024 against 0.6272 at N=10), but its nearest reference is `next_weaker`, at 0.1893 and 0.1802, beyond the threshold. Under random seating `next_weaker` no longer always shoots left, so it is a strength rule there rather than a direction.

Abstention behaves differently under the two seatings. At fifteen players every random seed misses a
little, between 0.0165 and 0.0230 of its decisions, while two of the three ranked seeds essentially never
miss and the third is step 1's holding-fire seed at 0.2976. At ten players one random seed misses in
0.0983 of its decisions. Stagnation is 0.0000 in every run, so missing never becomes a stand-off.

## What this establishes

Seating changes what the agent learns at every size measured. Under ranked seating the agent can read its
position from its neighbours, because accuracies ascend round the circle, and it learns a policy that
depends on rank. Under random seating the arrangement is reshuffled every episode and an agent cannot see
it, so the policy flattens to a rule that barely depends on rank. At five players that flat rule
coincided with `shoot_strongest`; at ten and fifteen it does not coincide with either reference.

## Reproducibility

Two random streams are reset, not one: the environment's, and torch's global stream, because a trained
policy samples its actions. The notebook measures the same thing twice and prints whether the results are
identical. They are.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook step6_seating.ipynb
```

This folder stands alone: it carries its own copy of the environment and its own pinned requirements, and
imports nothing from any other step.
