# Step 9 — Which way round the circle, and how often?

**Twenty seeds per player count, to measure how often each of the agent's two policies is learned and what the difference does.**

## Why this step exists

Step 8 found that a trained run settles on one of two mirror-image policies. Under ranked seating, always shooting left is `next_weaker`: each player fires at the next weaker one, and the weakest at the strongest. Always shooting right is its mirror, `next_stronger`: each player fires at the next stronger one, and the strongest at the weakest. Four of step 8's twenty runs at 10 and 15 players fired right. Steps 1 to 7 use seeds 0 to 2, which all fire left, so every result this folder reports past the answer key rests on the left-shooting policy without saying how common it is.

This step measures that rate, tests whether it depends on the player count, and compares what the two directions do: survival, exploitability on the switch test, targeting, and, at seven players, the exact answer.

## What it contains

- `train_and_measure.py`, step 8's script run with the centralised critic only: 3 player counts × 20 seeds, in parallel, each run trained, measured over 10,000 episodes, switch-tested as in step 2, and saved to `models/` with its record.
- `results/`, the sixty result files and `run_log.txt`.
- `step9_direction.ipynb`, which reads the files and does every comparison. It trains nothing.

Everything follows step 1: `ranked` seating, `terminal` reward, 2000 updates × 96 episodes, seeds 0 to 19.

## What number to expect

The script took 115.9 minutes with ten processes on an Apple M4 Pro, alongside two other notebooks: 6.0 to 10.2 minutes per run at N=7, 15.7 to 17.1 at N=10, and 29.2 to 34.9 at N=15. No run stalls.

**Seeds 0 to 4 reproduce step 8** exactly at 10 and 15 players, and at seven players all five fire left, as in deliverable 03 step 10.

**How a run is classified.** With the whole field alive, shooting left hits the stronger neighbour at 2 of the N seats and shooting right at N − 2, so the share of shots at the stronger neighbour reads near 2/N or near (N − 2)/N. Every run is near one or the other except one, seed 13 at N=15 (0.2597), which is more than 0.10 from both and is flagged as mixed; it is counted as left.

**The right-shooting policy is learned in about one run in eight, at every size:**

| N | right-shooting runs | rate | exact 95% interval |
|---|---|---|---|
| 7 | 2 of 20 (seeds 6, 17) | 0.10 | 0.0123 to 0.3170 |
| 10 | 2 of 20 (seeds 3, 4) | 0.10 | 0.0123 to 0.3170 |
| 15 | 4 of 20 (seeds 4, 8, 18, 19) | 0.20 | 0.0573 to 0.4366 |
| all | 8 of 60 | 0.13 | 0.0594 to 0.2459 |

A permutation test finds no dependence on the player count: 0.7071 of 20,000 shuffles give a spread of counts at least as large as the one observed.

**Each run sits on its direction's rule.** The left-shooting runs are on average 0.0848, 0.0493 and 0.0349 from `next_weaker`'s targeting by rank at N=7, 10 and 15, and the right-shooting runs 0.0266, 0.0943 and 0.1551 from `next_stronger`'s.

**What the direction does.** With two right-shooting runs at N=7 and N=10 the groups are too small to judge, so the comparison is made only at N=15, with sixteen left and four right:

- **The strongest player is the one that pays.** Under right-shooting the strongest seat survives 0.4819 times an even share, against 1.5288 under left-shooting, a difference of −1.0469 with a standard error of 0.2129. That follows from the rule: the strongest shoots the weakest, and the next strongest shoots the strongest.
- **The weakest player does not measurably differ** (1.6193 against 1.5077, standard error of the difference 0.1964), and neither does exploitability on the switch test (+0.0024, standard error 0.0021).

At N=7 and N=10, the right-shooting runs' weakest players survive 0.6751 and 1.0270 times an even share against 1.2413 and 1.4327 for left-shooting, but with two runs each that is description, not a measured difference.

**At seven players, against the exact answer.** No run of either direction reaches the gate. The eighteen left-shooting runs have worst ratios of 0.465 to 0.877 and sit 0.1128 from the equilibrium's targeting on average; the two right-shooting runs reach 0.446 and 0.812 and sit 0.5434 from it, about as far as `next_stronger` itself (0.5508). So the right-shooting policy is further from the equilibrium, as well as rarer.

**Among the left-shooting runs, the endgame decides survival of the weakest.** A run that keeps to `next_weaker` to the very end hits the stronger neighbour 1/2 of the time with four left and 2/3 with three. Runs that do so ("on the rule", within 0.05 at both) give the weakest player about what `next_weaker` gives it; runs that depart from it near the end give it much more:

| N | on the rule | departs | weakest × even share, on the rule | departs | `next_weaker` |
|---|---|---|---|---|---|
| 7 | 3 runs | 15 runs | 0.900 | 1.310 | 0.821 |
| 10 | 2 runs | 16 runs | 0.990 | 1.488 | 1.014 |
| 15 | 6 runs | 10 runs | 1.153 | 1.720 | 1.179 |

The difference is +0.410 (standard error 0.043) at N=7 and +0.567 (0.053) at N=15; at N=10 there are only two runs on the rule. Across all left-shooting runs, the share of shots at the stronger neighbour with three left correlates with the weakest player's advantage at 0.880, 0.722 and 0.642 at N=7, 10 and 15.

## What this establishes

**The agent's policy past the answer key is not one policy but three outcomes of the same training.** About seven runs in eight fire left round the circle, as `next_weaker`; about one in eight fires right, as its mirror `next_stronger`; and among the left-shooting runs, most depart from the rule near the end of the game by turning on the stronger neighbour, and those are the runs in which the weakest player does best. Which outcome a run reaches is a property of its seed: the rate of right-shooting does not measurably change with the player count, and step 8 found both directions under both critics.

So the reported survival of the weakest past the answer key is an average over these outcomes, and the endgame is what moves it. Steps 1 to 7, with three seeds, saw only left-shooting runs; their conclusions about the left-shooting policy stand, and this step adds how often it is the one learned.

**What it does not establish.** Twenty seeds give a rate to within about ±0.15 at each count. The comparison between directions rests on four right-shooting runs at N=15 and two at the other counts. Everything is under ranked seating, the `terminal` reward and one training budget.

## Reproducibility

Every run in the script seeds itself, so any one can be re-run alone and gives the same result file and the same model weights. The notebook runs a shortened job twice through the same code and prints whether the two results are identical. They are.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train_and_measure.py --workers 10     # about two hours with ten cores; resumable
jupyter notebook step9_direction.ipynb
```

This folder stands alone: it carries its own copy of the environment and its own pinned requirements, and imports nothing from any other step.
