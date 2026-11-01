# Step 1 — The same agent, at 10 and 15 players

Deliverable 03 trained this agent at 2, 3, 5 and 7 players and could mark its homework every time,
because folder 02 computes the exact answer at those counts. This step takes the identical agent to **10
and 15 players**, where no exact answer is on record.

**What is new here is 10 and 15.** Counts 3, 5 and 7 are re-measured at the same budget so the trend is
one experiment rather than five results stitched together, and because they still have an answer key
they check that this folder's copy of the machinery behaves.

## How much code had to change

**One line, and it was a deletion.** Step 9 of deliverable 03 looks up the known correct answer for each
player count in order to grade against it. There is no known answer at 10 or 15, so that lookup comes
out. Nothing else — not the network, not the environment, not the training loop, not the measurement.

| | at N=3 | at N=15 |
|---|---|---|
| action space | `Discrete(3)` | `Discrete(3)` |
| observation | 4 numbers | 4 numbers |
| actor | 4 → 3, 515 parameters | 4 → 3, 515 parameters |
| critic input | 9 | 45 |
| critic parameters | 705 | 3009 |

The actor is the same size at every count, which is why one trained actor could be run at any count. The
critic grows, and it is discarded after training.

## What changing the player count actually moves

More than one thing, which the step prints rather than glossing:

| N | accuracy gap | decisions per episode | critic input | stagnation limit | hard cap |
|---|---|---|---|---|---|
| 3 | 0.2000 | 3.7 | 9 | 30 | 600 |
| 5 | 0.1000 | 7.4 | 15 | 50 | 1000 |
| 7 | 0.0667 | 11.1 | 21 | 70 | 1400 |
| 10 | 0.0444 | 16.7 | 30 | 100 | 2000 |
| 15 | 0.0286 | 25.8 | 45 | 150 | 3000 |

**The accuracy range is pinned to 0.4–0.8 whatever the count**, so a larger field is also a tighter one —
the gap between neighbouring players is seven times smaller at 15 than at 3. A fifteen-player game is
not a three-player game with twelve more players in it; it is a game in which everyone is nearly equally
skilled. [`../GAME.md`](../GAME.md) sets out the arithmetic.

## What number to expect

`ranked` seating, `terminal` reward, 3 seeds, 2000 updates × 96 episodes = 192,000 training
episodes per seed, 10,000 measured episodes per seed. The whole step runs in **about 50 minutes** on a laptop CPU; wall-clock time varies from run to run.

### Where an answer key still exists

| N | measured, weakest rank first | worst ratio | reaches 0.95 |
|---|---|---|---|
| 3 | 0.3700 / 0.3335 / 0.2965 | 0.999 | yes |
| 5 | 0.2490 / 0.2129 / 0.1857 / 0.1578 / 0.1947 | 0.974 | yes |
| 7 | 0.1706 / 0.1459 / 0.1308 / 0.1218 / 0.1176 / 0.1230 / 0.1903 | 0.841 | **no** |

The three verdicts match deliverable 03's: pass at 3 and 5, fail at 7. **That is a check on this folder's
machinery rather than independent corroboration** — the training configuration and seeds are identical to
step 9's, so the models are the same models and only the measurement sampling differs. The two sets of
numbers agree to within sampling error at every rank. What this would have caught is a transcription
error in the copy-forward, which is what it is for.

### Where no answer key exists

```
N=10   0.1503  0.1309  0.1111  0.1014  0.0904  0.0809  0.0740  0.0704  0.0741  0.1164
N=15   0.1011  0.0921  0.0843  0.0773  0.0713  0.0645  0.0626  0.0571  0.0524  0.0497
       0.0480  0.0429  0.0434  0.0547  0.0985
```

**Both are U-shaped.** The weakest player does best, the field falls away through the middle to a minimum
at rank 7 of 10 and rank 11 of 15, and then the strongest players recover — the strongest seat takes
0.1164 at ten players and 0.0985 at fifteen. That non-monotone shape is what the exact answer looks like
at 5 and at 7, and it is still here where nothing can check it.

The step asks for the comparison at all five counts and **prints the refusal** at 10 and 15: *"no exact
answer is on record at N=10, so there is nothing to compare against; above the recorded range the method
changes from validating to characterising."* The other two preconditions are exercised on the same real
measurement, so the refusals are refusals of a real request rather than hypothetical.

## The trend, and what it is not

The weakest player's survival as a multiple of an even $1/N$ share:

| N | agent | per seed | exact | `shoot_strongest` | `coin_flip` |
|---|---|---|---|---|---|
| 3 | 1.110 | 1.108 / 1.104 / 1.117 | 1.111 | 1.118 | 0.647 |
| 5 | 1.245 | 1.206 / 1.215 / 1.315 | 1.248 | 2.022 | 0.685 |
| 7 | 1.194 | 0.928 / 1.296 / 1.357 | 1.073 | 2.983 | 0.666 |
| 10 | 1.503 | 1.555 / 1.445 / 1.510 | none on record | 4.442 | 0.692 |
| 15 | 1.517 | 1.173 / 1.731 / 1.646 | none on record | 6.769 | 0.710 |

**The agent's ratio does not rise at every step and `shoot_strongest`'s does.** At 15 players the agent
sits at **0.22** of that heuristic's advantage. So whatever the agent is doing, it is not reproducing the
signature of concentrating fire on whoever is strongest — which is the thing a steeply climbing ratio
would otherwise suggest.

**A rising ratio is not a result.** Over the range where the exact answer exists it goes 1.111, 1.248 and
then *down* to 1.073, while the heuristic's climbs at every step. So the direction of this column is a
property of the policy producing it, and the agent's own numbers cannot settle which policy that is —
two different policies can produce the same survival profile. That is what the targeting measurement in
[`../step3-targeting/`](../step3-targeting/) is for.

## The result that was not expected: one seed in three plays differently at 15 players

Abstention — the brief's second phenomenon, and the one the classical "the weakest should miss" result is
about:

| N | abstention, pooled | per seed |
|---|---|---|
| 3 | 0.0002 | 0.0003 / 0.0000 / 0.0002 |
| 5 | 0.0001 | 0.0001 / 0.0003 / 0.0000 |
| 7 | 0.0001 | 0.0000 / 0.0002 / 0.0000 |
| 10 | 0.0003 | 0.0007 / 0.0001 / 0.0000 |
| **15** | **0.0992** | **0.0000 / 0.2976 / 0.0001** |

**That is not a trend, it is a split.** At 15 players one seed abstains at a rate of 0.2976 and the other
two essentially never do. The pooled figure of 0.0992 describes no seed in the experiment, which is
the same trap the targeting diagnostic was built to avoid: an average across a bimodal outcome.

Three things keep this honest. **No episode ever ended in mutual abstention** — the stagnation rate is
0.0000 at every count including that seed, so abstaining was frequent without ever being coordinated.
**The seed spread in survival is widest at the same two counts**, 7 and 15, where the per-seed weakest
ratio runs 0.928 / 1.296 / 1.357 and 1.173 / 1.731 / 1.646 — against 1.108 / 1.104 / 1.117 at three
players, where the three seeds agree to a thousandth. The notebook prints two standard errors per rank for this reason. The evaluation's binomial error is about 0.002 everywhere, but the seed-to-seed error is far larger where the seeds disagree: up to 0.0324 at seven players and 0.0296 at fifteen, against 0.0016 or less at three. The seed error is the honest error bar on a pooled number. And **three seeds is
enough to see this but not enough to characterise it** — what the step establishes is that at 15 players
the training outcome is not reproducible in kind, not how often each kind appears.

### When it holds fire, and who does

The notebook breaks down every run that abstains in more than 1% of its decisions. Only one does: N=15,
seed 1.

**It holds fire at the start and shoots once the field thins.** With all 15 players alive it misses in
0.7808 of its decisions, then 0.6256 at 14, 0.4209 at 13, 0.2356 at 12 and 0.1022 at 11. By eight alive
it is 0.0032, and from four alive down it never misses at all. **0.991** of its misses are made while
more than two thirds of the field remains. That is a coherent pattern — wait while the others shoot each
other — rather than noise spread evenly through the game.

**Almost everyone does it.** Every rank misses between 0.1842 and 0.3492 of the time, so it is not one
kind of player. The weakest misses less than the ranks just above it (0.2421 against 0.3399 and
0.3492). Players between their two neighbours in strength miss most (0.3545); players weaker than both
(0.1900) or stronger than both (0.1845) miss about half as often.

**The exact answer has something similar in timing but not in who.** At eight players the exact
equilibrium holds fire in exactly one state, the opening one, and only two strong players do it. This
run also holds fire mostly at the opening, but nearly every player takes part. That is a resemblance,
not a match: it is a different player count, and at 15 players nothing can compute the exact answer to
say whether waiting actually pays.

This matters beyond this folder. At every smaller player count the agent essentially never abstains,
which reads as the agents selecting the shooting equilibrium. At 15 players, with this budget, one seed
in three did not.

## What this step hands to the next one

A survival profile at 10 and 15 players that nothing can check against an exact answer. One test still
works without one: folder 02 defines an equilibrium as a strategy no single player can improve on by
changing what it does alone, and that can be measured directly by letting one seat switch to something
else. That is the next step.

## Reproducibility

Two random streams are reset, not one: the environment's, and torch's global stream, because a trained
policy **samples**. The notebook ends by measuring the same thing twice and printing whether the results
are identical. They are.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook step1_train_any_n.ipynb
```

This folder stands alone: it carries its own copy of the environment and its own pinned requirements, and
imports nothing from any other step.

**Next:** [`../step2-switch-test/`](../step2-switch-test/)
