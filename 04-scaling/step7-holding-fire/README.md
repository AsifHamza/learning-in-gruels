# Step 7 — How often does an agent hold fire at 15 players?

Step 1 trained three seeds at fifteen players. Two essentially never abstained; one held fire in 0.2976
of its decisions, almost all of them early in the game. Three seeds can show that this happens but not
how often. This step trains ten, so the share of seeds that hold fire becomes a rate with an error bar.

Nothing else changes: `terminal` reward, `ranked` seating, step 1's budget. Seeds 0 to 2 are step 1's own
seeds, so they reproduce its numbers exactly; seeds 3 to 9 are new.

A run **holds fire** if more than 0.01 of its decisions are deliberate misses, the same threshold step 1
used to decide which runs to break down.

## What number to expect

`terminal` reward, `ranked` seating, 2000 updates × 96 episodes per run, seeds 0 to 9, 10,000 measured
episodes per seed. Ten training runs, about 8 minutes each and 79 minutes in all, with four notebooks
running at once on a laptop CPU; wall-clock time varies.

| seed | abstention | weakest/uniform | holds fire? |
|---|---|---|---|
| 0 | 0.0000 | 1.173 | no |
| 1 | 0.2976 | 1.731 | yes |
| 2 | 0.0001 | 1.646 | no |
| 3 | 0.0000 | 1.610 | no |
| 4 | 0.0004 | 1.731 | no |
| 5 | 0.0003 | 1.827 | no |
| 6 | 0.0000 | 1.548 | no |
| 7 | 0.0001 | 1.662 | no |
| 8 | 0.0037 | 1.083 | no |
| 9 | 0.0000 | 1.155 | no |

**One seed of ten holds fire, a rate of 0.10.** The exact binomial (Clopper-Pearson) 95% interval is
0.0025 to 0.4450, which is wide. It rules out a rate of one half or more, so holding fire is not the usual outcome, but it does not rule out it being common: step 1's rate of 1 in 3 lies inside it. Stagnation and the hard cap are 0.0000 in every seed, so no episode ended in a stand-off.

**The one seed that holds fire is step 1's seed 1, and it does so in the same pattern.** It misses 0.7808
of the time with all fifteen alive, 0.6256 with fourteen, 0.4209 with thirteen, and almost never once the
field has thinned: 0.991 of its misses come while more than two thirds of the field remains. None of the
seven new seeds shows it.

## What this establishes

Holding fire at fifteen players is a real outcome of training, seen in one seed of ten. Ten seeds do not separate a rate of one in ten from step 1's one in three, since both lie inside the interval. Where it happens it is concentrated at the opening of the game.
Ten seeds are enough to say it is not the typical outcome and not enough to say how rare it is.

## Reproducibility

Two random streams are reset, not one: the environment's, and torch's global stream, because a trained
policy samples its actions. The notebook measures the same thing twice and prints whether the results are
identical. They are.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook step7_holding_fire.ipynb
```

This folder stands alone: it carries its own copy of the environment and its own pinned requirements, and
imports nothing from any other step.
