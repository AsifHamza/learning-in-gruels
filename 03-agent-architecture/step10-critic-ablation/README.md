# Step 10 — Is the centralised critic needed?

**The same agent trained twice, changing only what the critic is given.**

> **This folder stands alone.** It has its own copy of `gruel_env.py`, its own
> `requirements.txt`, and imports nothing from any other folder. Copy it anywhere and it runs.

Player counts here: **N=3, 5 and 7**.

## Why this step exists

Steps 3 to 9 use MAPPO: a shared actor that sees four numbers, and a **centralised critic** that sees every seat's accuracy and alive flag plus a one-hot saying whose value it is estimating. Step 3 said the extra player at N=3 made that critic necessary, and nothing measured it. It is the first question a reviewer will ask, and the MAPPO paper this project cites (Yu et al. 2022) reports that independent PPO, where each agent's critic sees only what that agent sees, is often about as good.

So this step runs the ablation. The **local** arm is independent PPO with parameter sharing: the critic is given the same four numbers as the actor. Everything else is held fixed: the actor, the seeds, the budget, the update, the `terminal` reward, `ranked` seating and the measurement. The actor layers are built before the critic, so with the same seed both arms start from the same actor weights.

The local critic is not blind at every size. At N=3 with all three alive its four numbers are the whole state. At N=5 and N=7 it cannot see who sits beyond the two neighbours, so that is where a difference should appear.

## What it contains

- The step 9 network with the critic's input as a parameter, and one function, `critic_input_for`, that is the only place the arms differ.
- A check that the centralised arm reproduces step 9 seed for seed.
- Thirty runs: three player counts, two arms, five seeds, at step 9's budget.
- The gate per seed, a paired comparison of worst-rank ratios, the pooled survival profile, targeting against the exact equilibrium, and the explained variance of each critic.

## What number to expect

About 54 minutes on a laptop CPU; wall-clock time varies.

**The centralised arm is step 9.** Its seeds 0 to 2 give worst ratios of 0.986 / 0.992 / 0.986, 0.952 / 0.953 / 0.961 and 0.693 / 0.888 / 0.820 at N=3, 5 and 7, the same as step 9 to three decimals, so the ablation changed nothing but the critic. Every run in both arms has stagnation 0.0000.

**The gate verdict is the same in both arms at every size:**

| N | centralised, worst ratio per seed | passing | local, worst ratio per seed | passing |
|---|---|---|---|---|
| 3 | 0.986 / 0.992 / 0.986 / 0.990 / 0.990 | 5 of 5, PASS | 0.987 / 0.990 / 0.975 / 0.964 / 0.979 | 5 of 5, PASS |
| 5 | 0.952 / 0.953 / 0.961 / 0.927 / 0.971 | 4 of 5, PASS | 0.972 / 0.975 / 0.972 / 0.982 / 0.979 | 5 of 5, PASS |
| 7 | 0.693 / 0.888 / 0.820 / 0.459 / 0.883 | 0 of 5, FAIL | 0.475 / 0.678 / 0.854 / 0.300 / 0.692 | 0 of 5, FAIL |

**Paired by seed, the worst-rank ratio, centralised minus local:**

| N | centralised mean | local mean | difference | s.e. |
|---|---|---|---|---|
| 3 | 0.989 | 0.979 | +0.010 | 0.005 |
| 5 | 0.953 | 0.976 | -0.023 | 0.008 |
| 7 | 0.748 | 0.600 | +0.149 | 0.047 |

The rule fixed before the run was that a difference counts if it exceeds twice its standard error, and all three do. Read them by size, though. At N=3 and N=5 the differences are 0.010 and 0.023, in **opposite** directions, and neither changes a verdict. At N=5 the local arm is the more consistent one: all five seeds clear 0.95, where the centralised arm has three seeds within one evaluation standard error of the line and one below it. With five seeds a two-standard-error bar is loose (four degrees of freedom), so these two small differences should not be leaned on.

**At N=7 the centralised critic helps, and it is not enough.** The gap is 0.149, about three standard errors, and it shows in targeting too: P(shoot the stronger neighbour) by rank is 0.0739 from the exact equilibrium's on average for the centralised arm and 0.1092 for the local arm. The local arm's middle ranks fire at the stronger neighbour 0.3965 / 0.3618 / 0.3555, where the equilibrium has 0.2517 / 0.2088 / 0.2533 and the centralised arm 0.2496 / 0.2209 / 0.2348. Neither arm passes a single seed.

**The N=7 advantage is not an artefact of firing direction.** Folder 04 step 8 found that a run can learn the mirror-image policy, firing right round the circle instead of left, and that this depends on the seed rather than the critic. So the notebook classifies each run by its share of shots at the stronger neighbour with the whole field alive: near 2/N for a left-shooter, near (N−2)/N for a right-shooter. All ten runs at N=5 fire left. At N=7 one run fires right, the local arm's seed 3 (0.9969), which is also its worst seed (0.300). On the four seeds where both arms fire left, the centralised critic is still better, by +0.146 with a standard error of 0.060. With four seeds that is a modest margin, but it is the same size as the gap over all five.

**The centralised critic does not predict the return better.** Explained variance on a fresh 200-episode batch, $1 - \operatorname{Var}(G - V)/\operatorname{Var}(G)$, is 0.115 against 0.120 at N=3, 0.177 against 0.191 at N=5 and 0.200 against 0.221 at N=7, centralised against local. Both are low because a `terminal` return is a single win-or-lose outcome that the start of a game says little about. So whatever helps the centralised arm at N=7, it is not a critic that is better on average across all decisions.

## What this establishes

**Where the agent works, the centralised critic is not what makes it work.** At N=3 and N=5 independent PPO with a local critic passes the same gate, recovers the same survival profile (ratios 0.993 to 1.019 at N=5) and the same targeting (0.0063 from exact at N=5, against 0.0050). **Where the agent fails, at N=7, the centralised critic gets it measurably closer and does not rescue it.** So the case for MAPPO in this project is that it is no worse where both work and better at the size where the problem is hardest, not that the extra player made it necessary. Step 3's wording is corrected to say so.

What it does not establish: whether a larger budget would close the N=7 gap for either arm. Whether the advantage holds at larger N was tested afterwards in folder 04 step 8, without an answer key: at 10 and 15 players the switch test finds no measurable difference between the critics.

## Reproducibility

Both random streams are reset, the environment's and torch's, because a trained policy samples its actions. The notebook measures the local arm twice and prints whether the results are identical. They are.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook step10_critic_ablation.ipynb
```

## Where this sits

- Previous: [`../step9-player-count/`](../step9-player-count/): the same agent at N=3, 5 and 7.
- The whole investigation: [`../README.md`](../README.md)
