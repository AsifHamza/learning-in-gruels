# Step 6 — The N=5 gate

**The first real pass or fail, against a profile that is not monotone.**

> **This folder stands alone.** It has its own copy of `gruel_env.py`, its own
> `requirements.txt`, and imports nothing from any other folder. Copy it anywhere and it runs.

Player count here: **N=5**.

## Why this step exists

**Why N=5 and not N=3.** At N=3 the exact answer coincides with a simple heuristic, so
matching it proves very little. At N=5 the exact profile has a shape no heuristic produces:

| Accuracy | 0.4 (weakest) | 0.5 | 0.6 | 0.7 | 0.8 (strongest) |
|---|---|---|---|---|---|
| Exact survival | **0.24959** | 0.21073 | 0.17970 | **0.16194** | 0.19804 |

The weakest does best, and the **second-strongest fares worst of anyone** — worse than the strongest, and
worse than the middle. **An agent producing a tidy monotone ordering has failed this step**, however
reasonable that ordering looks.

**Why ranked seating.** The exact profile above is computed for one specific arrangement: accuracies
ascending around the circle. Comparing a random-seating run against it would be comparing against the
wrong number. Step 7 then changes exactly that.

## What it contains

- Training MAPPO at N=5 under ranked seating with the terminal reward.
- The ratio gate from step 4 applied at all five ranks.
- Several seeds, because one seed is an anecdote.
- The targeting diagnostic from step 5, now at an N where it has something to show.

## What number to expect

**`GATE: PASS`** — the rule is *reaches 0.95 on the worst rank, in at least 3 of 5 seeds*, and the
measurement is **4 of 5**. The one seed that misses reaches `0.927`, and it misses in an informative
direction: it *exaggerates* the non-monotone shape rather than flattening it, giving the weakest `0.2626`
against an exact `0.24959` and driving accuracy 0.7 down to `0.1501`.

**The PASS is at the threshold, not clear of it.** Each seed's worst ratio comes from 10,000 evaluation episodes, so it has a standard error of about 0.017 to 0.022. Seeds 0, 1 and 2 pass at `0.952`, `0.953` and `0.961`, all within one standard error of 0.95; only seed 4 (`0.971`) is clear of the line by more than one. So evaluation noise alone could move the count from 4 of 5 to below the 3 the rule needs. The gate is reported as passed because that is what the rule says, and read as "the agent reaches the threshold", not "it beats it".

**The five seeds pooled, against the exact answer:**

| accuracy | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 |
|---|---|---|---|---|---|
| measured | `0.2475` | `0.2107` | `0.1856` | `0.1593` | `0.1969` |
| exact | 0.24959 | 0.21073 | 0.17970 | 0.16194 | 0.19804 |
| ratio | 0.992 | 1.000 | 1.033 | 0.984 | 0.994 |

**The ordering is recovered exactly, in every seed.** All five produce
`[0.7, 0.6, 0.8, 0.5, 0.4]` worst-surviving first — identical to the exact ordering. So the answer to
the question this step exists to ask is **yes, the worst place to be is 0.7**, five times out of five.

**Targeting shows the U-shape, in all five seeds.** The extremes measure `1.0000` while the three
middle ranks sit between `0.2995` and `0.4066`. The test applied is deliberately strict — the *lower* of
the two extremes must exceed the *highest* of the three middle ranks, so a shape holding only on average
would not count.

**And targeting rises as the circle contracts, in all five seeds** — for seed 0, `0.4033` at five
alive, `0.5035` at four, `0.9665` at three, against an exact equilibrium of 0.40 / 0.50 / 1.00.

**Neither the U-shape nor the rise is enough on its own.** Under `ranked` seating the circle stays
sorted as players die, so Archetti's rule restricted to the two neighbours (shoot the strongest
neighbour still weaker than you; if both are stronger, the stronger one) is the same as always
shooting left. That fixed rule hits the stronger neighbour at two of the $k$ seats, giving 2/5, 2/4
and 2/3 by players left, and it is U-shaped by rank too (folder 04 step 3 measures it). What separates
the agent from it is the size of the numbers: the agent's middle ranks sit at `0.2995` to `0.4066`,
where Archetti's rule gives 0.2013 / 0.0930 / 0.2038 and the equilibrium 0.3979 / 0.3107 / 0.3714
(folder 02 step 5), and with three left the agent fires at the stronger neighbour `0.9665` of the
time, against 2 in 3 for the fixed rule. On those two counts the agent has reproduced the
equilibrium's **policy**, not merely the survival numbers that policy produces.

**`stagnation rate: 0.0000`** in every seed, with abstention at `0.0001` or below — so the targets
describe the game actually being played.

## What to watch in the output

**Whether the worst rank is 0.7.** The single most informative line in the notebook. If the
minimum sits at 0.8 instead, the agent has found a monotone policy and missed the equilibrium.

**That the gate is a necessary condition and not a sufficient one.** `shoot_strongest` passes a
survival gate at N=3 while getting the targeting policy wrong from N=4 onward. Items 3, 4 and 5 of the
notebook's closing summary are the non-monotone minimum, the U-shape, and the rise with contraction.
The U-shape and the rise are shared with Archetti's rule on the circle (see above); what no heuristic
reproduces is the minimum at 0.7 together with the size of the middle ranks and the near-certain
shot at the stronger neighbour with three left.

**That there are two random streams, not one.** The reproducibility check reseeds the environment
*and* `torch`, because a trained policy samples its actions. Reseeding only the game leaves the policy's
stream where the previous call left it, and the check then fails while nothing is actually wrong. Every
earlier step happened to test a deterministic heuristic, so this only surfaces here.

## Reproducibility

The notebook ends by running the same thing twice from the same seed and printing whether the
two results are identical. A step that cannot reproduce itself cannot support a claim.

## Where this sits

- Previous: [`../step5-targeting/`](../step5-targeting/) — The diagnostic that survival numbers cannot replace.
- Next: [`../step7-seating/`](../step7-seating/) — One dial moved. Same agent, same player count, different answer.
- The whole investigation: [`../README.md`](../README.md)
