# Step 2 — PPO at N=2

**The same game as step 1, so the only thing that changes is the algorithm.**

> **This folder stands alone.** It has its own copy of `gruel_env.py`, its own
> `requirements.txt`, and imports nothing from any other folder. Copy it anywhere and it runs.

Player count here: **N=2**.

## Why this step exists

**Why the game does not change here.** Holding N=2 fixed while replacing the algorithm is
deliberate: if the numbers move, the algorithm moved them. Change the game at the same time and you learn
nothing about either.

**Why PPO over REINFORCE.** Two additions, each fixing something specific.

A **value baseline**: REINFORCE weights every action by the whole episode return, including the part that
was going to happen regardless. Subtracting a learned estimate of that leaves only the part the action was
responsible for, which is a far less noisy signal.

A **clipped objective**: REINFORCE uses each batch once and discards it. PPO takes several gradient epochs
on the same batch, and the clip is what makes that safe — it refuses updates that move the policy so far
that the collected data no longer describes it. Self-play episodes are the expensive part, so using each
batch several times matters.

## What it contains

- An actor-critic network: two small networks, one choosing actions and one scoring positions.
  They deliberately **share no weights**. Step 3 gives the critic the whole game state while the
  actor still sees only its own four numbers, and once the two take different inputs a shared trunk
  is impossible — so keeping them apart here makes step 3 a smaller change. Many real
  implementations do share a trunk.
- The advantage, and why it is the return minus the value estimate.
- The clipped ratio: first as a worked table showing why the clip is asymmetric, then measured on
  real transitions as the **share of ratios falling outside the clip window** on every update.
- Several epochs over one batch, which REINFORCE cannot do.
- A budget sweep against REINFORCE at the same budgets and seeds, so the efficiency question is measured rather than asserted.

## What number to expect

The same survival answer as step 1: the stronger player wins two duels in three. The step was written expecting PPO to reach it from **far fewer episodes** than REINFORCE. Measured against REINFORCE at the same budgets, it does not, and on this game it cannot be expected to.

Both algorithms are trained from scratch at each budget on seeds 0 to 3 and evaluated over 20,000 episodes. The table gives the weak player's ratio to exact, as the mean over the four seeds with the lowest seed in brackets.

| Episodes of self-play | PPO | REINFORCE |
|---|---|---|
| 320 | 0.981 (0.972) | 0.987 (0.981) |
| 640 | 0.983 (0.977) | 0.988 (0.984) |
| 1280 | 0.986 (0.976) | 0.984 (0.981) |
| 2560 | 0.987 (0.983) | 0.989 (0.983) |

One standard error of a ratio from the evaluation alone is 0.010, so no row differs from any other by more than evaluation noise, and both algorithms reach 0.97 on every seed at the smallest budget, 320 episodes. **The duel is too easy to show a difference in sample efficiency.** Its optimal policy is "always shoot", which both algorithms find within a few hundred episodes.

An earlier version of this README claimed PPO matched step 1 "from 9.4 times fewer episodes". That compared PPO's 640 episodes with the 6,000 that step 1 happened to use, without asking how few REINFORCE needed. Trained for 640 episodes, REINFORCE gives 0.990 on seed 0, the same as at 6,000. The claim is withdrawn.

PPO is kept for what the later steps need from it rather than for speed here. The critic it adds is what step 3 turns into a centralised critic that sees the whole game state, and the clip is what lets each batch of self-play be reused for several epochs. Whether that reuse pays off is a question for games harder than the duel, and this project does not measure it.

The headline run uses 2,560 episodes, because a longer run draws a more readable learning curve. It
measures **0.3277** for the weak player and **0.6723** for the strong one, ratios of 0.983 and 1.008.

## What to watch in the output

**How often the clip actually binds.** If it never binds, the updates are small enough
not to need it. If it binds constantly, the learning rate is too high. Measured here: **0.3617** of
ratios fall outside the window on the first update, settling to a mean of **0.0471** across the run.
That shape is what you want — the clip does real work while the policy is moving fastest, then
mostly stops mattering.

**The loss going negative here is not an error.** Step 1's loss could never be negative — it was
$-\log \pi \times R$ with a reward that is never negative. Here the return has the value estimate
subtracted from it, and that advantage is negative whenever an episode went worse than the critic
expected. Measured: **22 of 40** updates end on a negative policy loss.

**But not until the critic is trained**, and the notebook shows why. Untrained, the critic predicts
between −0.4534 and −0.4127 for every position — an arbitrary number from the random
initialisation, and a *negative* one, so $R - V(s)$ is positive even when the return is zero and
**0.000** of advantages come out negative. Trained, it predicts between 0.5669 and 0.8432, so every
decision in a lost episode scores negative and the share rises to **0.175**. An untrained baseline
subtracts nothing useful; that is the whole reason the critic has to be fitted.

## Reproducibility

The notebook ends by running the same thing twice from the same seed and printing whether the
two results are identical. A step that cannot reproduce itself cannot support a claim.

## Where this sits

- Previous: [`../step1-reinforce/`](../step1-reinforce/) — The smallest possible policy gradient, on the smallest possible game.
- Next: [`../step3-mappo/`](../step3-mappo/) — The centralised critic. This is the architecture the capstone is about.
- The whole investigation: [`../README.md`](../README.md)
