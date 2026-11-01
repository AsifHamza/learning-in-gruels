# learning-in-gruels — the Gruel project

Start here. This is the map.

Every result in the project is produced by the code in these folders, built up in small pieces at a
level that can be read line by line.

---

## What the project is about

A **gruel** is an N-player elimination duel on a graph. N players sit on a circle, each with a fixed
**accuracy** — the chance a shot they fire hits. One player is chosen at random each step and may
shoot their left neighbour, shoot their right neighbour, or deliberately miss. You may only shoot
someone **adjacent** to you. When a player dies, the circle **contracts**: their two former
neighbours become neighbours of each other. Last player standing wins.

The result the project is about is **survival of the weakest**: the worst shot is the most likely to
survive, because the good shots spend their turns shooting each other.

At N=3 with accuracies 0.8 / 0.6 / 0.4, the exact survival probabilities are

| Accuracy | Survival | Fraction |
|---|---|---|
| 0.8 | 0.29630 | 8/27 |
| 0.6 | 0.33333 | 9/27 |
| 0.4 | **0.37037** | 10/27 |

Those three numbers are the thread running through everything here. Folder 01 measures them by
simulation, folder 02 computes them exactly, and folder 03 trains agents that have to rediscover
them.

### Which game this is, and which published results belong to it

Results in the truel literature are easy to compare across games that differ in turn order or in
whether a player may abstain. This repository implements **random turn order, abstention allowed, on a
circle that contracts at each death**. The table says which published result comes from which game.

| Game | Published result | Source |
|---|---|---|
| Three players; order drawn at random once, then fixed; abstention allowed | the weakest may do best by firing into the air | Kilgour 1975 |
| Three players; random order; no abstention | the 8/27, 9/27, 10/27 triple above, under the stronger-opponent strategy | Kilgour and Brams 1997, p. 317 |
| N players; random order; no abstention; anyone may shoot anyone | survival of the weakest; shoot the type immediately weaker, while the weakest shoots the strongest | Amengual and Toral 2006; Archetti 2012 |
| N players; fixed order; no abstention; anyone may shoot anyone | a stationary equilibrium exists for every N, computed by recursion | Mastrakoulis and Kehagias 2025 |
| Three players; random order; abstention allowed | shooting and non-shooting equilibria can coexist | Kehagias and Mastrakoulis 2026 |
| On a graph; fixed order; no abstention | the gruel defined; on the four-player star the stronger-opponent strategy is optimal | Lanphier 2017 |

No paper in the table studies this repository's game, which is why folder 02 computes its own exact
answers. Two consequences follow.

**"The weakest should miss" does not apply here.** It is a result about the fixed-order game. In this
game the shooting equilibrium never abstains: Kehagias and Mastrakoulis (Prop. 4.5) show that in the
random-order truel with abstention allowed, a player always gains by shooting and targets the same
opponent Kilgour and Brams priced. That is also why the 8/27, 9/27, 10/27 triple, which comes from a game
without abstention, is a valid target for this one. Their non-shooting equilibrium (Prop. 4.6) needs a
cost to being eliminated, which the win-only reward does not have.

**Survival of the weakest is caused by targeting, not by missing.** It appears in games where nobody may
miss at all, so reproducing it never requires an agent to abstain.

Wegener and Mutlu (2021) report that survival of the weakest reverses on scale-free networks, but their
model is a different game: fixed order, no abstention, a network that only samples three-player
encounters, and eliminated players replaced rather than the graph contracting. It is a contrast, not
something this repository can reproduce.

The papers themselves, with each one's rules set out in full, are in
`Capstone Project/04 Execution/Papers` of the course vault, summarised in `Research Compendium.md` there.

---

## The four deliverables

Read them in order. Each folder has its own README, and each **step** inside it has its own README
too.

| Folder | Deliverable | What it is | Status |
|---|---|---|---|
| [`01-environment/`](01-environment/) | **The game** | The rules in code: circle, contraction, turn order, shooting, rewards, four baseline policies, and a picture of it running | **done** |
| [`02-exact-solver/`](02-exact-solver/) | **The analytical answer** | Computes exact survival probabilities by working backwards from the end. No machine learning. Produces the **targets** | **done** |
| [`03-agent-architecture/`](03-agent-architecture/) | **The agent** | A neural network policy trained with MAPPO, seeing only its two neighbours. Built up from REINFORCE at N=2, then checked against the exact answers at N=3, N=5 and N=7 — passing at 3 and 5, failing at 7 — while varying seating, reward profile and player count one at a time | **done** |
| [`04-scaling/`](04-scaling/) | **Beyond the answer key** | The same agent at N=10 and N=15. Almost no new code — what changes is that the ground truth has run out, so the result has to be judged another way | **done** |

**The shape of it:** 01 and 02 build the *ground truth*. 03 builds an agent and checks it against that
ground truth at N=3, N=5 and N=7. 04 trains the same agent at N=10 and N=15, where no ground truth
exists, and judges it another way.

---

## Where the ground truth stops, and why that is the whole point

Folders 01–03 can all be checked against arithmetic. Folder 04 cannot, and that is not a weakness of
folder 04 — it is the reason the project trains agents at all.

| N | States to solve | Strategy combinations | Rough total checks | Exact answer |
|---|---|---|---|---|
| 3 | 4 | 27 | 108 | yes, instant |
| 5 | 26 | 243 | 6,318 | yes, ~0.004 s |
| 7 | 120 | 2,187 | 262,440 | yes, ~0.1 s |
| 8 | 247 | 6,561 | 1,620,567 | yes |
| 9 | 502 | 19,683 | 9,880,866 | **no — no pure equilibrium exists** |
| 10 | 1,013 | 59,049 | 59,816,637 | **no — refused at a state with 8 players alive** |
| 15 | 32,752 | 14,348,907 | 469,955,402,064 | **no — refused at a state with 8 players alive** |

Timings measured in [`02-exact-solver/step3-circle-n5/`](02-exact-solver/step3-circle-n5/) and
[`step4-how-far/`](02-exact-solver/step4-how-far/); counts are arithmetic. Roughly a factor of
5 per player added.

**What stops the exact approach is not cost.** Run on 9, 10 and 15 players, the solver
refuses every time, and every time at a state with eight players alive; at 15 players it gets there in
about three minutes (`02-exact-solver/step4-how-far`, section 4.7). N=8 solves in under half a second, and
carrying the measured growth forward puts N=10 at around ten seconds, which is perfectly
affordable. The real wall is at
N=9, where a state has **no pure Nash equilibrium**, so backward induction over pure strategies
has nothing to select. That state does have a **mixed** equilibrium, in which two players
randomise between a shot and holding fire, so a solver over mixed strategies could go further;
this project does not build one. Past N=8 there is no exact answer on record, and the only way
to say anything is to train agents and describe what they do.

---

## For readers of the capstone report

The report states its methods and results; this repository holds the code, the saved models and every notebook with its outputs.

**Where each result in the report comes from.**

| Report section | Steps |
|---|---|
| The exact answers | `01-environment/step2`, `02-exact-solver/step2` to `step5` |
| Validation against the exact answers | `03-agent-architecture/step1` to `step6`, `step9`; `04-scaling/step3` |
| Is the centralised critic needed? | `03-agent-architecture/step10`; `04-scaling/step8` |
| Seating and reward at five players | `03-agent-architecture/step7`, `step8` |
| Behaviour at 10 and 15 players | `04-scaling/step1` to `step4` |
| Two policies, and the endgame | `04-scaling/step9` |
| Seating and reward at 10 and 15 players | `04-scaling/step5`, `step6` |
| Deliberate missing | `01-environment/step4`, `02-exact-solver/step4`, `04-scaling/step1`, `step6`, `step7` |
| Training Settings (appendix) | the records in each step's `models/*.json`, listed in [`MODELS.md`](MODELS.md) |

Each step's `README.md` states what it found and what number to expect, and its notebook is saved with its outputs.

**Seeds.** Every source of randomness is an explicitly seeded generator. For training seed $s$: torch is seeded with $2000 + s$, the training game with $1000 + s$ and the minibatch shuffle with $s$; the 10,000-episode measurement uses game seed $5000 + s$ with torch reset to $7000 + s$. The scripts in `04-scaling/step8` and `step9` seed each switch-test measurement on its own (baseline $11000 + s$, each switch $12000 + 1000s + 10 \times \text{seat} + \text{rule}$) and the critic's explained variance at $8000 + s$, so any one run reproduces alone. Every notebook ends by measuring the same thing twice from the same seeds and printing whether the results are identical. Exact reproduction of a training run is guaranteed only on the software versions pinned in each step's `requirements.txt`; the saved models make every published number re-measurable without retraining.

**Saved models and re-runs.**

- *Every trained model saved.* Each training step saves every model it trains with a record of its layers, parameter counts, inputs, every training setting and seed, software versions, and hashes of its weights and of the code that produced it. Because steps copy code forward rather than importing it, several train the same configuration; every such configuration has identical weights in every step that trains it, which checks the training pipeline across steps.
- *Re-runs compared.* Every training notebook was re-executed after the final changes, and its printed output compared line by line with the previous run's; every measured number was identical.

**Names in the report and in the code.** The report names the reference policies and reward profiles in words; the code and the step READMEs use the code names.

| Report | Code |
|---|---|
| stronger-neighbour rule | `shoot_strongest` |
| weaker-neighbour rule | `shoot_weakest` |
| random targeting | `coin_flip` |
| always missing | `always_miss` |
| next-weaker rule (Archetti's rule restricted to the two neighbours; under ranked seating, always shoot left) | `next_weaker` |
| next-stronger rule (its mirror; under ranked seating, always shoot right) | `next_stronger` |
| win-only reward | `terminal` |
| bonus reward | `shaped` |
| local critic (independent PPO) | `critic_view="local"`, arm `local` |
| centralised critic (MAPPO) | `critic_view="centralised"`, arm `centralised` |
| validation criterion (every rank at least 0.95 of exact in a seed, in a majority of seeds) | "gate" |
| seating and reward conditions | "the dials" |
| condition (one setting of a comparison) | "arm" |
| beyond the solver's range, where no exact solution exists (10 and 15 players) | "past the answer key" |
| a game stalls (with $k$ alive, $10k$ consecutive deliberate misses; any shot, hit or not, resets the count) | `stagnation`, `miss_streak`, `stagnation_threshold()` |

**Environment details the report leaves out.** The environment passes PettingZoo's own `api_test` (`01-environment/step3`). The hard cap of $200N$ steps exists to catch faults; no run in the project reached it.

---

## Where to start reading

If you have twenty minutes:

1. [`01-environment/step2-plain-game/`](01-environment/step2-plain-game/) — the entire game in one
   class, ending on 8/27, 9/27, 10/27.
2. [`02-exact-solver/step2-truel-n3/`](02-exact-solver/step2-truel-n3/) — the same three numbers
   derived exactly, then checked against a 1997 paper.

If you have an hour, run folder 01 end to end (about a minute of compute) and then folder 02.

If you only want the picture: [`01-environment/step5-render/`](01-environment/step5-render/), then
`../.venv/bin/python play.py --n 6 --step`.

---

## What is built, and verified

**All four deliverables are complete.** 29 notebooks (5 in folder 01, 5 in 02, 10 in 03, 9 in 04) are
executed and saved **with their outputs**, so they can be read without running anything.

| Check | Result |
|---|---|
| `01/step2` — 100,000 episodes, shoot-strongest | 0.3705 / 0.3324 / 0.2971 — worst 0.65 standard errors from exact |
| `01/step3` — PettingZoo's own `api_test` | passed |
| `01/step4` — four baselines and the focal-agent comparison | shoot-strongest beats every deviation at every accuracy |
| `02/step2` — the exact triple | 8/27, 9/27, 10/27 to 1e-15, by hand *and* by the general solver |
| `02/step2` — six published Kilgour & Brams rows | all six match; worst error 0.000471 |
| `02/step3` — N=5 circle, ranked seating | 0.19804 / 0.16194 / 0.17970 / 0.21073 / 0.24959, sums to exactly 1 |
| `02/step3` — 26 states, equilibrium multiplicity | unique equilibrium in every state |
| `02/step4` — how far the exact answers reach | solves N=3 to N=8; refuses at N=9, where a state has no pure equilibrium |
| `03/step1` — REINFORCE at N=2 against $p/(p+q)$ | 0.3301 / 0.6700 against 0.3333 / 0.6667 |
| `03/step2` — PPO at N=2, same answer | same accuracy as REINFORCE at every matched budget from 320 episodes; the duel cannot show a speed-up |
| `03/step3` — MAPPO at N=3, one seed | ordering correct, every rank within 3% of 8/27, 9/27, 10/27 |
| `03/step4` — instruments calibrated on heuristics | `always_miss` reports no data from 0 decisions, not a rate of 0 |
| `03/step6` — **the N=5 gate**, 5 seeds | **PASS** in 4 of 5, at the threshold (3 passing seeds within one evaluation SE of 0.95). Worst rank ordering recovered 5 of 5 |
| `03/step9` — N=3, 5, 7 against exact | N=3 and N=5 **pass** 3 of 3 seeds; **N=7 fails** 0 of 3 |
| `03/step10` — centralised critic against a local one (independent PPO) | same gate verdict at every N; centralised 0.149 better on the worst rank at N=7, where both fail |
| `04/step8` — the same comparison at 10 and 15, five seeds each | no measurable difference on the switch test; four of twenty runs learn the mirror-image right-shooting policy |
| `04/step9` — twenty seeds at 7, 10 and 15 | 8 of 60 runs fire right (rate 0.13, no dependence on N); among left-shooting runs the endgame decides the weakest player's advantage |
| `04/step2` — can one player gain by switching? | no profitable switch at N=5; at N=15 every seed has at least one |

### The trained models

Every training step saves the models it trains into its own `models/` folder, with a record of each:
layers and parameter counts, inputs, every training setting and seed, software versions and hashes.
[`MODELS.md`](MODELS.md) lists every model with its settings and hashes.

---

## The gates

A **gate** is a pass/fail checkpoint that must succeed before the project may advance. **A gate needs a
computable answer, so there are exactly two** — at N=3 and N=5, where the solver can produce the truth.
Above N=8 nothing can, so N=10 and N=15 are described rather than graded.

| Gate | Where the rule is applied here | Criterion | Status |
|---|---|---|---|
| **N=3** | `03/step9-player-count` | >=95% of 8/27, 9/27, 10/27 | **PASSED** — 3 of 3 seeds |
| **N=5** | `03/step6-n5-gate`, re-applied in `03/step9-player-count` | >=95% of the solver's ranked-seating profile | **PASSED** — 4 of 5 seeds in step 6, 3 of 3 in step 9 |

**Note that `03/step3-mappo` trains at N=3 but applies no gate.** It is where the architecture is
introduced, and the instrument that turns a ratio into a verdict does not exist until step 4. The first
pass/fail in the folder is step 6, and N=3 is graded in step 9 alongside N=5 and N=7.

**At N=7 the same rule fails**, 0 of 3 seeds, worst ratios 0.693 / 0.888 / 0.820. That is not a third
gate — nothing downstream was blocked on it — but it is the folder's most informative result and it is
reported in `03/step9-player-count`.

Both gates require: **outcome** metrics only (never miss frequency), the **`terminal`** reward profile
wherever an analytical target is involved, **shooting-class seeds only** with >=3 of 5 in that class, and
at N>=4 **ranked seating**, because that is how the solver's targets are computed. The last two are
enforced in code — the evaluator refuses to issue a verdict otherwise.

**Why there is no gate at N=10 or N=15.** Beating a baseline was considered and dropped as a criterion. A
large gap against `shoot_strongest` turns out to be that heuristic's own signature: where the truth is
computable the equilibrium's weakest-over-strongest ratio is nearly flat — 1.25x, 1.26x, 1.23x at N=3, 5
and 7, then inverts to 0.96x by N=8 — while `shoot_strongest`'s climbs at every step to 9.84x. A criterion rewarding a steep ratio
would reward resembling the heuristic. What is measured at those scales instead is **who the agent
shoots**.

---

## Layout

```
learning-in-gruels/
├── README.md                    this file
│
├── 01-environment/
│   ├── README.md                folder overview + index
│   ├── requirements.txt         one venv per phase, shared by its steps
│   ├── .venv/
│   ├── step1-circle-graph/      README + notebook
│   ├── step2-plain-game/
│   ├── step3-pettingzoo-env/
│   ├── step4-heuristics/
│   └── step5-render/            also holds play.py, the live window
│
├── 02-exact-solver/             README + 5 steps, all with executed notebooks
│   ├── README.md
│   ├── requirements.txt
│   ├── .venv/
│   ├── step1-duel/              N=2, and the cancellation the rest rests on
│   ├── step2-truel-n3/          N=3 by hand, then checked against six published rows
│   ├── step3-circle-n5/         N=5, where the graph starts to matter
│   ├── step4-how-far/           the walk upward: reaches N=8, refused at N=9
│   └── step5-equilibrium/       why the criterion works, and the targeting policy
│
├── 03-agent-architecture/       README + 10 steps, all with executed notebooks
│   ├── step1-reinforce/         no gate — REINFORCE at N=2
│   ├── step2-ppo/               no gate — PPO at N=2, same answer, no speed-up at matched budgets
│   ├── step3-mappo/             no gate — the architecture, at N=3
│   ├── step4-measurement/       no gate — builds the instruments, calibrated on heuristics
│   ├── step5-targeting/         no gate — who the agent shoots, calibrated at N=3 and N=5
│   ├── step6-n5-gate/           THE N=5 GATE, ranked seating
│   ├── step7-seating/           no gate — ranked against random
│   ├── step8-reward-profile/    no gate — terminal against shaped
│   ├── step9-player-count/      N=3, 5 and 7 against the exact answers — N=3 and N=5 pass, N=7 does not
│   └── step10-critic-ablation/  is the centralised critic needed? MAPPO against independent PPO
│
├── 04-scaling/                  README + GAME.md + 7 steps, all executed
│   ├── GAME.md                  the game end to end: counts, accuracies, contraction, rewards
│   ├── step1-train-any-n/       the agent at 10 and 15, and the five-count trend
│   ├── step2-switch-test/       can one player do better by switching? an equilibrium test
│   ├── step3-targeting/         who the agent shoots, read against the three signatures
│   ├── step4-continuity/        is it still the policy that worked at 7?
│   ├── step5-reward-profile/    terminal against shaped, at 10 and 15
│   ├── step6-seating/           ranked against random, at 10 and 15
│   ├── step7-holding-fire/      how often a run holds fire at 15, over ten seeds
│   ├── step8-critic-ablation/   is the centralised critic needed at 10 and 15? script + notebook
│   └── step9-direction/         how often each mirror-image policy is learned, over twenty seeds
│
└── MODELS.md                    every saved model, with its settings and hashes
```

Every step folder holds two files with the same content: the **`.ipynb`** to read, and a `.py` which
is the plain-text form of the same notebook, paired by jupytext. Ignore the `.py` unless you want to
diff it in git.

---

## Running anything

Each folder has its own virtualenv, shared by its steps.

```bash
cd 01-environment
python3 -m venv .venv                            # first time only
./.venv/bin/pip install -r requirements.txt      # first time only
./.venv/bin/jupyter lab
```

From inside a step folder, refer back up to it:

```bash
cd 01-environment/step2-plain-game
../.venv/bin/jupyter lab step2_plain_game.ipynb
```

Folder 02 needs almost nothing — its solver is plain Python with no numpy and no torch. Only Jupyter
and matplotlib are installed, and those are just for reading the notebooks.

---

## Six things these folders establish

Worth knowing before reading any code, because each one corrects an intuition.

**Survival-of-the-weakest is not a law of the game.** Of six published N=3 accuracy triples, three
do **not** show it. With 0.9 / 0.5 / 0.1 the best shot simply wins. The effect needs the accuracies
reasonably close together.

**It is caused by targeting, not by the rules.** Switch everyone to shooting their *weakest*
neighbour and the 0.4 player's survival collapses from 0.3705 to 0.0723.

**Nobody abstains in equilibrium.** Exactly one of the 27 possible strategy combinations at N=3 is
an equilibrium, and it never uses the deliberate-miss action. A lone abstainer survives 0.0000 of
the time. The project's original proposal expected the opposite and had to withdraw the criterion.

**Shoot-strongest is the equilibrium at N=3 and is not at N=5.** 18 profitable deviations exist at
N=5. The two coincide at N=3 only because a triangle is a complete graph.

**At N=5, survival is not monotone in accuracy** — the second-strongest does worst of anyone — and
that turns out to be caused by **seating**, not by the graph. Averaged over all 120 seatings it
disappears.

**The exact answers run out at N=8, and not for the reason the arithmetic suggests.** The work
does grow steeply — 26 states at N=5, 32,752 at N=15, needing about $4.7 \times 10^{11}$ strategy
checks — but the wall arrives earlier and for a different reason. At **N=9 a state has no pure Nash
equilibrium**, so backward induction over pure strategies has nothing to return. The state does
have a mixed equilibrium, so the limit is this method rather than the game; above N=8 there is no
exact answer on record, which is why the larger games are described rather than validated.
