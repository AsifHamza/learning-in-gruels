# Step 3 — The same game, wrapped in PettingZoo

## What this step is

Step 2's game works, so why wrap it in a framework?

Because the training code in folder 03 is not written from scratch — it comes from CleanRL, and
it expects an environment that looks a particular way: a standard method for resetting, a
standard way of asking whose turn it is, a standard place to find the reward. Speak that
language and existing training code can drive your game.

**PettingZoo** is the standard for multi-agent environments. Gymnasium is the single-player
equivalent, the one CartPole uses.

The game underneath is *exactly* step 2's game. Same rules, same numbers. Everything new here
is interface — and the point of the step is to see precisely where the line between "my game"
and "the framework" falls.

## How to run it

```bash
../.venv/bin/jupyter lab step3_pettingzoo_env.ipynb
```

About **15 seconds**, most of it the conformance test and the 50,000-episode check.

## What you should see

An episode driven by the standard PettingZoo loop:

```
accuracies by position: [0.8 0.4 0.6]

t= 1  player_0 plays shoot_left   -> alive: 2
t= 2  player_0 plays shoot_right   -> alive: 2
...
t= 8  player_0 plays shoot_right   -> alive: 1

ended because: sole_survivor
episode totals: {'player_0': 1.0, 'player_1': 0.0, 'player_2': 0.0}
```

That trace is **identical to step 2's** — same seed, same game, different interface.

PettingZoo's own conformance test:

```
Starting API test
Passed API test

api_test passed — this is a legitimate PettingZoo AEC environment.
```

You will also see one warning, `UserWarning: Observation numpy array is all zeros.` That is
expected — it is a dead agent's observation, which the environment deliberately returns as
zeros.

The survival numbers again, to prove the wrapper changed nothing:

```
  accuracy 0.4  ->  survives 0.3689
  accuracy 0.6  ->  survives 0.3336
  accuracy 0.8  ->  survives 0.2975

standard errors away from exact:
  accuracy 0.4: -0.67
  accuracy 0.6: +0.12
  accuracy 0.8: +0.59
```

Slightly different from step 2's numbers because it is a different number of episodes, but the
same answer within noise.

Seeding:

```
seed 0, run again: [0.8, 0.4, 0.4] [0.8, 0.4, 0.4]
same seed reproduces: True
```

If `api_test` fails, the environment is not a valid PettingZoo environment and nothing built on
top of it can be trusted.

## What to look at

**Why AEC and not the parallel API.** PettingZoo has two styles: everybody acts at once
(parallel), or one agent acts and then the environment updates (AEC — Agent Environment Cycle).
Our game is AEC and not by choice: only one player shoots per step and *who* is drawn at
random, so there is no round in which everybody acts. A parallel environment cannot express
that.

It costs us something real. Because agents act one at a time and at irregular intervals, there
is no single tidy sequence of state-action-reward for the whole game — each agent has its own
scattered history. That is the root of the hard part of folder 03, and it starts here.

**The dead step, section 3.3.** An agent does not vanish when it dies. It is marked terminated,
stays in `env.agents`, gets selected once more, and your loop must call `env.step(None)` for it
— *then* PettingZoo removes it. Forget the `None` and your loop hangs or throws; treat that
`None` step as a real decision and you feed garbage into training. It **fails quietly**, which
is the worst kind of failure.

**Neither reward dictionary is the episode total, section 3.4.** `rewards[agent]` is the most
recent step only. `_cumulative_rewards[agent]` is a running total but is reset to zero every
time that agent acts, because that is when the value is handed to the learner. So the class
keeps its own `episode_returns`.

**`api_test` checks things your own tests will not.** The class had no `close()`
method, and the test refused it: *"If render method defined, then close method required."* A
fair complaint — if an environment can open something, callers need a way to shut it — and one
no test of mine would have thought to make. Running someone else's test of your interface is
worth more than writing your own.

**`terminations` versus `truncations`.** Termination means the game ended by its own rules
(someone won, or it is stuck). Truncation means we cut it off from outside because it hit a step
limit. The payoff is identical either way; they are separated because a high truncation rate is
a **bug signal**, not a result.

## What it deliberately doesn't do

**No action masking.** All three actions are always legal here. Some environments tell the
agent which actions are available; ours does not need to, because you always have a left
neighbour, a right neighbour, and the option to abstain.

**No parallel-API version.** It could not express this game.

**No vectorisation.** Real training runs many environment copies at once for speed. Not here —
this is for reading.

**Next:** `step4-heuristics/` builds the four comparison policies and measures them properly,
with error bars and a way to compare policies when there are more than two players.
