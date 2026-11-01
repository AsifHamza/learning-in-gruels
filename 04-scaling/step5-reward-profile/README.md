# Step 5 — The reward profile, at 10 and 15 players

Deliverable 03's step 8 turned the reward dial at five players and found no difference in survival that
survived its error bars. It also made a prediction: the shaped profile's hints are worth what the kill
term can reach as a fraction of the win, which is 0.3 at five players, 0.8 at ten and 1.3 at fifteen. So
the same arithmetic predicts that if shaping is going to change anything, it should change it here. This
step tests that prediction at the two counts where no exact answer is on record.

One dial moves: `terminal` against `shaped`. Seating stays `ranked`, the reward is the only change, and
every other setting matches step 1.

## What number to expect

`ranked` seating, 2000 updates × 96 episodes per run, seeds 0 to 2, 10,000 measured episodes per seed.
Twelve training runs. About 77 minutes with four notebooks running at once on a laptop CPU; wall-clock
time varies.

The kill term a non-winner can bank, and whether it can out-pay a win:

| N | kill term a non-winner can bank | as a fraction of the win | net, if it then dies |
|---|---|---|---|
| 5 | 0.30 | 0.30 | -0.70 |
| 10 | 0.80 | 0.80 | -0.20 |
| 15 | 1.30 | 1.30 | +0.30 |

Even at fifteen players, where the kill term alone exceeds the win, a killer who then dies nets 0.30
against a winner's 1.0, so winning still pays most.

**Survival of the weakest, as a multiple of an even 1/N share:**

| N | terminal | shaped | difference | combined SE |
|---|---|---|---|---|
| 10 | 1.503 | 1.563 | +0.059 | 0.231 |
| 15 | 1.516 | 1.490 | -0.026 | 0.266 |

Neither difference clears twice its standard error, so there is **no measurable difference** in survival
at either count. The prediction that shaping would matter is not borne out in the survival numbers.

**Targeting, P(shoot the stronger neighbour), tells a sharper story.** At N=15 the two arms are the same
policy (mean distance 0.0052). At N=10 they differ (0.1243, just over the 0.10 threshold): the shaped
arm's U-shape is shallower, so it is the one place the reward changed what was learned, even though
survival did not move. Measured against `next_weaker` in the same game, the `terminal` arm matches it at
both sizes (0.0424 and 0.0225), and the `shaped` arm matches it at N=15 (0.0246) but not at N=10 (0.1665).

**No non-winner out-earned any winner** in 30,000 shaped episodes at either count, but that check could not have failed. The table above already guarantees it: a non-winner ends on at most the kill term minus 1, plus a survival tick of 0.001 a step, while a winner banks at least +1, so with episodes this short no non-winner can reach a winner. What the episodes add is where play sits against that ceiling. The best a non-winner banked was -0.283 at N=10 and -0.080 at N=15, against ceilings of -0.20 and +0.30 before the survival tick, and winners banked about +1.11. So non-winners came well short of the most the kill term allows.

Episode lengths stay short enough that the undiscounted survival tick is negligible: about 16 steps at
N=10 and 24 at N=15, so the tick adds at most 0.016 and 0.024 to a return.

## What this establishes

At the scale where shaping should bite hardest, it does not change which outcome the agent reaches. It
leaves a faint mark on *how* the agent plays at N=10 and none at N=15. Survival of the weakest is a
property of the game, not of the hints, at ten and fifteen players as at five.

## Reproducibility

Two random streams are reset, not one: the environment's, and torch's global stream, because a trained
policy samples its actions. The notebook measures the same thing twice and prints whether the results are
identical. They are.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook step5_reward_profile.ipynb
```

This folder stands alone: it carries its own copy of the environment and its own pinned requirements, and
imports nothing from any other step.
