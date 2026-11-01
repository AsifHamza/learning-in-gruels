# Step 8 — Game variant: the reward profile

**Does the finding survive the reward that produced it?**

> **This folder stands alone.** It has its own copy of `gruel_env.py`, its own
> `requirements.txt`, and imports nothing from any other folder. Copy it anywhere and it runs.

Player count here: **N=5**. Seating: **ranked**, held fixed. Two arms, three seeds each.

## Why this step exists

**Why this control is not optional.** Every result up to step 7 used the `terminal` reward — +1
for winning, nothing otherwise — because that is the objective folder 02's mathematics actually
solves. But training on it is hard: the signal arrives once, at the end, and only for one
player. `shaped` adds hints: a bonus for a kill, a penalty for being eliminated, a trickle for
staying alive.

**Why hints are dangerous.** They change what is being maximised. So if survival of the weakest
appears under `shaped` but not under `terminal`, it is an artefact of the hints rather than a
property of the game.

**And the reward is a different kind of dial from seating.** Seating changed the *game* — it
re-dealt the accuracies, so step 7's two arms were playing different situations. The reward
changes only the number handed to the optimiser. The notebook shows this rather than claiming
it: the same untrained policy from the same seed produces an **identical sequence of winners**
under both profiles, and the two heuristic references measure identical survival under both.
Any difference between the trained arms therefore arrived through learning and nothing else.

## What it contains

- The two profiles read out of the environment, not restated.
- The arithmetic of what a non-winner can bank, and the same arithmetic at N=3, 10 and 15.
- A check that the reward actually reaches the optimiser — see below, this is where the
  defect was.
- The same agent trained under each profile, with survival and targeting side by side.
- Which arm may legitimately be compared to the exact answer, enforced in code.

## The defect this step exposed

`collect_batch` computed the return for every decision as **1.0 for the winner and 0.0 for
everyone else**. That is exactly right under `terminal`, which is why steps 3 to 7 were
unaffected — but it never read the environment's rewards at all, so under `shaped` every kill
bonus, elimination penalty and survival tick was discarded. Both arms would have trained on
identical numbers, and the step would have reported "the reward changed nothing" as a finding
when it was a bug.

The return is now the **reward-to-go**: what a player ended the episode with, minus what it had
already banked when it decided. The notebook checks this before spending any compute — under
`terminal` the distinct returns are exactly **0.0 and 1.0**, which is also what steps 3 to 7
trained on, while under `shaped` there are 61 distinct values spanning **−0.9990 to +1.4070**.

## Can hunting out-pay winning?

A non-winner can bank at most $N-2$ kills, because killing $N-1$ opponents makes it the sole
survivor. At N=5 that is 0.3, against the −1.0 it is charged for being eliminated.

| bound | value |
|---|---|
| best a non-winner can do, kills and death only | **−0.7000** |
| the same, allowing every survival tick the episode cap permits | **+0.3000** |
| worst a winner can do — a win with no kills | **+1.0000** |
| best non-winner actually measured, 30,000 episodes | **−0.6850** |
| worst winner actually measured | **+1.1040** |

The measured best non-winner sits **0.0150** above the kills-only bound, which is 15 survival
ticks — a player that took every kill it could and stayed alive a while before dying. No
non-winner out-earned any winner.

**This is where a commonly quoted figure needs care.** At N=15 a non-winner can reach
$13 \times 0.1 = 1.3$, which exceeds the win reward of 1.0 **on the kill term alone** — and
that is sometimes read as showing shaping makes losing pay better than winning. It does not,
because the elimination penalty is certain: a player holding those kills and still losing is
dead by definition, and nets 0.3 against a winner's 1.0. What shaping changes is the
**gradient** — a kill is a certain, immediate +0.1 while the win is distant and uncertain — not
the ordering of outcomes.

## What number to expect

**The effect present under both profiles, and at this player count no measurable difference
between them.**

| | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | weakest ÷ uniform |
|---|---|---|---|---|---|---|
| `terminal` | 0.2431 | 0.2125 | 0.1855 | 0.1610 | 0.1979 | 1.215 |
| `shaped` | 0.2422 | 0.2148 | 0.1917 | 0.1618 | 0.1896 | 1.211 |

The difference is **−0.005 against a combined standard error of 0.031**, or 0.14 standard
errors, so it is not a direction. Per seed the weakest gets 1.188 / 1.190 / 1.268 under
`terminal` and 1.196 / 1.243 / 1.194 under `shaped` — the arms overlap.

**A null result here was predictable, and the notebook predicts it before measuring.** The
hints are worth what the kill term can reach as a fraction of the win: 0.10 at N=3, **0.30 at
N=5**, 0.80 at N=10, 1.30 at N=15. At N=5 they are a minor correction to the objective and
should barely move anything. At N=15 the fraction is 1.30, so the same reasoning predicts that
shaping matters there; this step tests the prediction only at N=5.

## And the policy did not change either

Survival numbers alone cannot say whether two arms learned the same thing — step 7 is the
demonstration, where survival said "random seating helps the weakest" and what had actually
happened was that the agent found a different policy. So the arms are compared **to each
other** on targeting:

| | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | spread |
|---|---|---|---|---|---|---|
| `terminal` | 0.9999 | 0.3790 | 0.3142 | 0.3745 | 0.9998 | 0.6857 |
| `shaped` | 0.9978 | 0.3734 | 0.3387 | 0.3987 | 1.0000 | 0.6613 |

Mean distance between the arms: **0.0113**, well inside the 0.10 threshold. Both show the
equilibrium's U-shape, and both track its contraction pattern against the exact 0.40 / 0.50 /
1.00 — `terminal` at 0.4038 / 0.5102 / 0.9763 (worst gap 0.0237) and `shaped` at 0.3997 /
0.5373 / 0.9667 (worst gap 0.0373).

**Note that both arms match `neither` heuristic reference**, at distances of 0.2873 and 0.2783.
That is why the arms are compared directly: two policies can each be unlike a coin flip and
unlike `shoot_strongest` while being unlike each other too, so "both matched neither" would
have said nothing about whether they matched one another.

## What to watch in the output

**That only the `terminal` arm is measured against the exact answer.** The exact solver
maximises survival probability; under `shaped` the agent maximises something else, so a
comparison against the exact numbers would be grading it on an objective it was never given.
`compare_to_exact` raises `WrongVariant` unless the reward is `terminal`, and the notebook asks
for the comparison on both arms so the refusal is visible. The terminal arm's ratios are 0.974
/ 1.009 / 1.032 / 0.994 / 0.999.

The refusal costs something real: **the shaped arm cannot be validated against ground truth at
all, at any N, however well it plays.** That is the price of the hints, and it is why every
gate in this folder is a `terminal` measurement.

Also worth watching: stagnation is **0.0000** in all six runs, so both arms are playing the
shooting equilibrium and the survival numbers mean what they appear to mean.

## Why the returns are undiscounted

Returns here are **undiscounted** in both arms. The usual reason to discount under `shaped` is
specific: with a discount of 0.99 and a survival tick of 0.001 the discounted stream caps at 0.1,
keeping it from rivalling the win reward. The notebook checks whether that cap is anywhere near binding at
this scale — mean episode length is **7.46 steps**, so the trickle accrues about 0.007, under a
hundredth of the win reward, against a cap of a tenth it never approaches. Discounting would
change a typical return by about a percent and add machinery whose only purpose is to bound
something already small.

## Reproducibility

The notebook ends by running the same measurement twice from the same seed and printing whether
the results are identical. **Both** random streams are reset — the environment's and torch's —
because a trained policy samples its actions, so reseeding only the environment leaves the
policy's stream where the previous call left it.

## Where this sits

- Previous: [`../step7-seating/`](../step7-seating/) — One dial moved. Same agent, same player count, different answer.
- Next: [`../step9-player-count/`](../step9-player-count/) — N=3, then 5, then 7 — and what has to change at N=8.
- The whole investigation: [`../README.md`](../README.md)
