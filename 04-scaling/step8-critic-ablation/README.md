# Step 8 — Does the centralised critic matter past the answer key?

**The same agent trained with two critics at 10 and 15 players, compared without an answer key.**

## Why this step exists

Deliverable 03's step 10 trained the agent twice, once with the centralised critic (MAPPO) and once with a critic given only the actor's four numbers (independent PPO). Both critics passed the same gates at N=3 and N=5, and at N=7, where both failed, the centralised critic did measurably better. The open question was whether that advantage holds at the sizes this folder studies. A report that argues for MAPPO needs the answer.

There is no exact answer at 10 or 15 players, so there is no gate. The comparison uses the measures this folder already uses where nothing can check the agent:

1. **The switch test from step 2.** For each seat, does switching alone to one of four simple rules raise its chance of winning? A profitable switch proves the policy is not an equilibrium. Two numbers per run: the **count** of profitable switches out of 4N tried, and the **exploitability**, the best gain any switch gave each seat, averaged over the seats.
2. **Survival of the weakest**, as a multiple of an even share.
3. **Targeting**, read against `next_weaker` from step 3.
4. **The critic's explained variance**, as in deliverable 03 step 10.

Only the switch test and the explained variance have a direction that means "better" without an answer key. The others are reported as higher or lower.

## What it contains

- `train_and_measure.py`, a script that trains and measures all twenty runs (2 player counts × 2 critics × 5 seeds) in parallel, one process per run, and writes one result file per run into `results/`. It is a script because the work runs for more than an hour in sequence. Every run seeds itself, and a run whose file exists is skipped, so an interrupted job can be resumed.
- `results/`, the twenty result files and `run_log.txt`.
- `step8_critic_ablation.ipynb`, which reads the files and does every comparison. It trains nothing and runs in under a minute.

The agent and critic code are deliverable 03 step 10's; the measurement is step 1's and the switch test step 2's, with one change: every switch measurement now seeds its own game, rule and torch streams, so each number reproduces on its own and runs can be done in any order. Everything else follows step 1: `ranked` seating, `terminal` reward, 2000 updates × 96 episodes, seeds 0 to 4, 10,000 measured episodes. The switch test uses 10,000 baseline episodes, 5,000 per switch, and a three-standard-error threshold.

## What number to expect

The script took 38.9 minutes with ten processes on an Apple M4 Pro: about 12 minutes per run at N=10 and 24 to 29 at N=15.

**The centralised arm is step 1.** Its seeds 0 to 2 reproduce step 1's survival by rank exactly at both player counts. No run stagnates or hits the cap.

**First, a finding about the agent rather than the critic: it learns one of two mirror-image policies.** Under ranked seating, always shooting left (`next_weaker`) hits the stronger neighbour at 2 of the k seats and always shooting right at k − 2 of them. With the whole field alive the share of shots at the stronger neighbour therefore reads near 2/N for a left-shooter and near (N − 2)/N for a right-shooter:

| N | 2/N | (N−2)/N | centralised, per seed | local, per seed |
|---|---|---|---|---|
| 10 | 0.2000 | 0.8000 | 0.1947 / 0.2013 / 0.1973 / 0.8034 / 0.8039 | 0.7962 / 0.1970 / 0.1974 / 0.2000 / 0.2030 |
| 15 | 0.1333 | 0.8667 | 0.1304 / 0.1191 / 0.1380 / 0.1358 / 0.9321 | 0.1304 / 0.1341 / 0.1315 / 0.2539 / 0.1304 |

So four of the twenty runs fire right: seeds 3 and 4 of the centralised arm and seed 0 of the local arm at N=10, and seed 4 of the centralised arm at N=15. Both directions appear under both critics, so which one a run lands on is a property of the seed, not of what the critic sees. One run, the local arm's seed 3 at N=15 (0.2539), sits between the two and is counted as left. Steps 1 to 4 used seeds 0 to 2, which all fire left; seeds 3 and 4 show that the left-shooting policy step 3 describes is the common outcome at these sizes, not the only one.

**The switch test finds no measurable difference between the critics.** Paired by seed, centralised minus local:

| N | measure | centralised, per seed | local, per seed | difference | s.e. |
|---|---|---|---|---|---|
| 10 | profitable switches, of 40 | 1 / 1 / 2 / 4 / 7 | 0 / 0 / 3 / 0 / 1 | +2.2 | 1.2410 |
| 10 | exploitability | 0.0008 / -0.0015 / 0.0057 / 0.0104 / 0.0179 | 0.0032 / -0.0021 / 0.0025 / -0.0022 / 0.0009 | +0.0062 | 0.0037 |
| 15 | profitable switches, of 60 | 6 / 7 / 0 / 3 / 1 | 6 / 4 / 8 / 8 / 1 | -2.0 | 1.9748 |
| 15 | exploitability | 0.0102 / 0.0090 / 0.0001 / 0.0047 / 0.0045 | 0.0102 / 0.0035 / 0.0085 / 0.0085 / 0.0025 | -0.0010 | 0.0024 |

None clears twice its standard error, and the sign flips between N=10 and N=15. Restricted to the seeds where both arms fire left, there is still no measurable difference at N=15, and only two such seeds at N=10, too few to judge. What the switch test does separate is **direction**: at N=10 the seven left-shooting runs have a mean exploitability of +0.0006, against +0.0105 for the three right-shooting runs, which include the two largest switch counts, 7 and 4. Most of the centralised arm's apparent disadvantage at N=10 comes from its two right-shooting seeds.

**The profitable switches are in the same place for both critics.** At N=15, both arms' profitable switches are almost all made by seats 9 to 13, the strong players short of the strongest, and mostly to `shoot_strongest`, as step 2 found.

**The weakest player's advantage does not differ measurably:** 1.3128 against 1.5226 at N=10 and 1.5780 against 1.5729 at N=15, centralised against local, with standard errors of the difference of 0.2281 and 0.2197. The N=10 centralised mean is pulled down by seed 3, a right-shooter whose weakest player survives 0.4580 times an even share.

**The local critic predicts the return slightly better**, as at N=3 to 7 in deliverable 03: explained variance 0.2150 against 0.2312 at N=10 and 0.2113 against 0.2289 at N=15, differences of -0.0161 and -0.0177 with standard errors of 0.0052 and 0.0053.

**At N=15 seed 0 the two arms play identically.** Both actors settle on the same nearly deterministic left-shooting policy, and with shared seeds every sampled game is the same, so every number they produce is equal except the critic's own explained variance (0.2298 against 0.2513). That seed contributes a difference of zero to every behavioural comparison, which is the right answer for it rather than an artefact.

## What this establishes

**Past the answer key, the centralised critic shows no measurable benefit on any measure taken here, and its value estimates are slightly worse.** The advantage it showed at N=7 in deliverable 03 step 10 does not appear at 10 or 15 players on the switch test, survival or targeting. What does change the result is the direction a run happens to fire, which both critics produce and which no critic setting controls.

So the case for MAPPO in this project rests on N=7 alone: no worse at N=3 and N=5, better at N=7, and no measurable difference at 10 and 15.

**What it does not establish.** The switch test tries four simple rules, so "no profitable switch" is evidence, not proof, of an equilibrium, and the comparison is only as sensitive as those four rules are. With five seeds per arm, a difference of about two profitable switches per run is below what the test can detect here. And all of this is under ranked seating and the `terminal` reward.

## Reproducibility

Every run in the script seeds itself, so any one of them can be re-run on its own and gives the same file. The notebook runs a shortened job twice through the same code and prints whether the two results are identical. They are.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train_and_measure.py --workers 10     # about 40 minutes with ten cores; resumable
jupyter notebook step8_critic_ablation.ipynb
```

`python train_and_measure.py --n 10 --arm local --seed 0` runs one job alone.

This folder stands alone: it carries its own copy of the environment and its own pinned requirements, and imports nothing from any other step.
