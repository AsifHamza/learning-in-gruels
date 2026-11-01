# Step 9 — Player count

**N=3, then 5, then 7 — the same agent, three sizes of field, each with an exact answer to check against.**

> **This folder stands alone.** It has its own copy of `gruel_env.py`, its own
> `requirements.txt`, and imports nothing from any other folder. Copy it anywhere and it runs.

Player counts here: **N=3, N=5 and N=7**.

## Why this step exists

**Why player count comes last.** Every earlier step holds N fixed and moves something else. This one
moves the size of the field, which changes the game more than any other dial: the circle is longer, the
contraction has further to go, and each player's two neighbours are a smaller share of the whole. It
belongs after the method is established because it is the step that asks whether the method's answer
was about the game or about one size of it.

**Why these three.** Because an exact answer is on record for each, so all three can be checked rather
than merely described. The recorded answers reach N=8, so this step is comparing against ground truth
at every point and is not standing at the boundary.

**The one thing to get right.** Three player counts give three exact answers, and they do not line up
in a straight line. Reporting three separate results would miss the finding; reporting a straight-line
trend would invent one.

## What it contains

- The same agent, same architecture, same `terminal` reward, same `ranked` seating, trained at N=3, N=5
  and N=7 — three seeds each, nine runs.
- Survival of the weakest relative to an even share at each N, in one table.
- The exact answer beside each, from the recorded reference.
- `shoot_strongest` measured at all three, because it is the alternative explanation and it makes a
  *different* prediction.

## The agent, in full

**This is the terminal folder of deliverable 03, so the complete agent is here** — not imported from
anywhere, not split across steps. `step9_player_count.py` holds the network, the training loop and the
measurement; `gruel_env.py` holds the game. Those two files are the whole of it.

### The environment is a PettingZoo AEC environment

`class GruelEnv(AECEnv)` implements the standard surface — `reset()`, `observe()`, `step()`, `last()`,
`agent_iter()`, `observation_space()`, `action_space()` — so the training loop is the ordinary PettingZoo
one:

```python
for acting_agent in game.agent_iter():
    observation, reward, termination, truncation, info = game.last()

    if termination or truncation:
        game.step(None)
        continue

    actor = game.position_of_agent[acting_agent]
    ...
    game.step(int(action))
```

**Three things about that loop are easy to get wrong.** `agent_iter()` hands back an agent that has
already finished, once, so the environment can retire it — stepping with `None` is how you say *nothing
to do here*, and omitting it hangs the iteration. `observe()` takes an agent **name**, not a position.
And the actor is **drawn at random with replacement** rather than cycling, so turn order is not
round-robin and a player can act twice in a row.

### One network, two heads

```python
class SharedActorCritic(nn.Module):
    def __init__(self, n_players):
        super().__init__()
        self.n_players = n_players
        self.actor_first = nn.Linear(OBSERVATION_SIZE, HIDDEN_SIZE)
        self.actor_second = nn.Linear(HIDDEN_SIZE, N_ACTIONS)
        critic_input_size = 2 * n_players + n_players
        self.critic_first = nn.Linear(critic_input_size, HIDDEN_SIZE)
        self.critic_second = nn.Linear(HIDDEN_SIZE, 1)
        self.activation = nn.Tanh()
```

One model per run, **shared by every player** — the players are not separate learners, they are the same
parameters acting from different seats. Two paths through it:

| | Input | Shape | Hidden | Output |
|---|---|---|---|---|
| **Actor** | one player's own observation | `[4]` — **fixed, whatever N** | 64, Tanh | `[3]` logits → softmax → sample |
| **Critic** | the whole game state plus who is being scored | `[3N]` — 9, 15, 21 at N=3, 5, 7 | 64, Tanh | `[1]`, the value |

**The asymmetry is the architecture, not an accident.** The actor's input never mentions N, so the same
trained actor runs unchanged at any player count. The critic's input scales with N, so it cannot. That
is **centralised training with decentralised execution**: the critic sees everything while learning and
is then thrown away, and what ships is a policy that reads four numbers.

**The actor reads a single observation** and nothing from any other player.

### What the critic is given, and why it is `V(s, i)` rather than `V(s)`

`global_state()` walks the positions and emits, for each, its accuracy and then a flag for whether it is
still alive — `2N` numbers. `one_hot()` appends which position is being scored — `N` more. Hence `3N`.

The one-hot is load-bearing. Because the actor is drawn at random, a critic given only the state cannot
tell **whose** return it is being asked to predict, and two players in the same state have very different
prospects. Conditioning on the agent is what makes the value estimate answerable.

### The training loop

`collect_batch()` plays episodes and records, per decision, the observation, the action, its log
probability, the value estimate, the global state and the agent one-hot. The **return** for a decision is
reward-to-go: the player's final episode total minus whatever it had already banked when it decided. That
is read from `game.episode_returns`, so it is the environment's own arithmetic rather than an assumption
about when rewards arrive.

`ppo_update()` is textbook PPO: normalise the advantages, form the ratio of new to old action
probabilities, clip it, take the pessimistic branch, add a squared-error value loss and subtract an
entropy bonus, then run several epochs of shuffled minibatches over the same batch.

| Setting | Value | |
|---|---|---|
| updates × episodes per update | 2000 × 96 | 192,000 training episodes per seed |
| learning rate | 0.003 | Adam |
| clip | 0.2 | the PPO trust region |
| epochs × minibatch | 4 × 256 | how often one batch is reused |
| value / entropy coefficients | 0.5 / 0.01 | |
| evaluation episodes | 10000 | exploration still on, because the policy samples |
| seeds | 3 | per player count |

### What is deliberately absent

**No graph network**, as above. **No memory** — the network is feedforward and the observation carries no
history, so an agent cannot recall who shot at whom; what it knows of the past arrives only through its
neighbours' current accuracies and the survivor fraction. **No discounting**, and under this
reward profile that is exact rather than approximate. Every reward in `terminal` is zero except `win`,
which is `1.0`, so an undiscounted reward-to-go is precisely the win indicator: `1.0` for every decision
the eventual winner made, `0.0` for everyone else. The critic's output is therefore an estimated
**survival probability** directly, which is what makes this profile the one comparable against an
analytic answer. Discounting would break that correspondence by shrinking the win in proportion to how
long the episode happened to run.

## The three exact answers are three different shapes

This is what makes three player counts worth training rather than one. Survival probabilities,
weakest rank first:

| N | exact equilibrium survival, weakest rank first | best off | worst off |
|---|---|---|---|
| 3 | 0.37037 / 0.33333 / 0.29630 | rank 0, the weakest | rank 2, the strongest |
| 5 | 0.24959 / 0.21073 / 0.17970 / 0.16194 / 0.19804 | rank 0, the weakest | rank 3 |
| 7 | 0.15335 / 0.17343 / 0.14916 / 0.13761 / 0.12996 / 0.13217 / 0.12432 | **rank 1** | rank 6, the strongest |

**The ordering changes with the field size.** At N=3 it runs straight down. At N=5 the weakest is still
best off but the *second-strongest* is worst, so there is a kink. At N=7 the weakest is **no longer even
the best-off player**, and the worst seat is the strongest one. An agent that produces the same shape at
all three has not tracked the exact answers, however sensible that shape looks.

## What number to expect

**The gate is step 6's rule, applied unchanged at each count:** every rank's measured survival must
reach 95% of its exact target.

| N | pooled verdict | seeds reaching 0.95 | worst ratio per seed |
|---|---|---|---|
| 3 | **PASS** | 3 of 3 | 0.986 / 0.992 / 0.986 |
| 5 | **PASS** | 3 of 3 | 0.952 / 0.953 / 0.961 |
| 7 | **FAIL** | 0 of 3 | 0.693 / 0.888 / 0.820 |

Measured against exact, pooled over 30,000 episodes per count:

| N | measured, weakest rank first | ordering recovered |
|---|---|---|
| 3 | 0.3678 / 0.3354 / 0.2968 | yes |
| 5 | 0.2431 / 0.2125 / 0.1855 / 0.1610 / 0.1979 | yes |
| 7 | 0.1671 / 0.1456 / 0.1309 / 0.1218 / 0.1206 / 0.1237 / 0.1904 | **no** |

**N=7 fails in a specific and readable way.** Its ratios against exact run 1.089 / 0.839 / 0.878 / 0.885
/ 0.928 / 0.936 / 1.531 — **every middle rank sits below both extremes**, which is not true at N=3 or
N=5. The agent over-serves the two ends of the circle and under-serves everyone between them, and the
strongest seat gets more than one and a half times its exact share.

**The N=5 arm agrees with two earlier steps to four decimals.** Step 7's `ranked` arm and step 8's
`terminal` arm are the same configuration as this step's N=5 arm, and all three measure 0.2431 / 0.2125 /
0.1855 / 0.1610 / 0.1979. Three folders, three separate copies of the environment, three separate training
runs, one profile — which is what keeping a separate copy in each folder buys.

## The headline, and why it is three rows and not one

The weakest player's survival as a multiple of an even 1/N share. These are **three different
quantities** — the first is arithmetic, the other two are measurements — so a slope read down a column
that mixed them would belong to none of them.

| row | N=3 | N=5 | N=7 |
|---|---|---|---|
| exact equilibrium | 1.111 | 1.248 | 1.073 |
| the agent | 1.104 | 1.215 | 1.169 |
| `shoot_strongest`, measured | 1.118 | 2.022 | 2.983 |
| `coin_flip`, measured | 0.647 | 0.685 | 0.666 |

**The two explanations predict opposite directions between N=5 and N=7** — the exact answer falls by
0.174, `shoot_strongest` climbs by 0.961. The agent changes by −0.046 against a combined standard error
of 0.147, which is 0.3 standard errors. **So neither prediction can be selected**, and the notebook says
so rather than naming the nearer one.

## What the agent actually did at N=7

**It learned a structured policy, not nothing.** P(shoot the stronger neighbour) by rank is U-shaped at
N=7 with a spread of 0.8513 — *more* pronounced than N=5's 0.6857 — with the extremes near 1 and the
middle down to 0.1486. At N=3 the profile is flat and high (0.9965 to 0.9995), which is **correct
there**: at N=3 the equilibrium *is* `shoot_strongest`, so there is no U-shape to find.

**Compared at equal survivor counts, the N=5 and N=7 runs agree at five and four alive and differ at three:**

| players alive | N=5 run | N=7 run | difference |
|---|---|---|---|
| 5 | 0.4038 | 0.4031 | 0.0007 |
| 4 | 0.5102 | 0.5081 | 0.0021 |
| 3 | 0.9763 | 0.8419 | 0.1343 |

**Matching at equal survivor counts is weak evidence on its own.** Folder 04 step 3 shows that under ranked seating a fixed rule, always shooting the left-hand neighbour, gives 2/5 and 2/4 at five and four alive, so two runs that both stay close to that rule agree there whatever they learned. The informative comparison is with the exact N=7 answer by rank (folder 02 step 5):

| rank, weakest to strongest | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| agent, N=7 | 0.9994 | 0.3142 | 0.1743 | 0.1486 | 0.1852 | 0.3196 | 0.9999 |
| exact, N=7 (rounded) | 1.00 | 0.47 | 0.25 | 0.21 | 0.25 | 0.54 | 0.80 |

**That is why the ratio did not come down.** Every middle rank fires at its stronger neighbour less often than the equilibrium does, and the gap is largest at the top of the ranking: the second-strongest at 0.3196 against about 0.54, and the strongest at 0.9999 against about 0.80. In the equilibrium the second-strongest fires at the strongest more often than not, and the strongest sends about a fifth of its shots at the weakest player beside it; the agent's second-strongest mostly fires the other way, and its strongest fires at the second-strongest almost every time. Its strongest seat then survives more than twice its exact share (0.2579 against 0.12432 in the probe below). "The effect keeps growing" and "the agent did not change what it was doing" predict the same flat ratio, and only the per-rank profile separates them.

**A budget probe, one seed and twice the updates, does not show undertraining, and cannot rule it out.** The worst ratio moves from 0.693 to 0.713 while the gate needs 0.95, and the strongest seat stays at more than twice its exact share (0.2579 → 0.2536 against an exact 0.12432). The gain of 0.020 is within the seed-to-seed spread of the worst ratio at 2000 updates (0.693 / 0.888 / 0.820 across the three seeds), so one seed at a doubled budget cannot distinguish a small gain from noise. Doubling did move the targeting further from the equilibrium, not closer: the probe's middle ranks fall to 0.2619 / 0.1060 / 0.0816 / 0.1204 / 0.2781. Reported as a probe rather than a gate.

## What to watch in the output

**Whether each N matches its own exact answer at every rank.** That is the gate, and it is what carried
the finding here: the ratio alone would have reported "roughly flat from N=5 to N=7" and stopped.

**That the trend is read only across rows measuring the same thing.** Exact answers and measured runs are
different quantities.

**That the classification happens before any survival number is read.** All nine runs measured 0.0000
stagnation and 0.0000 hard-cap, so all nine are playing the shooting equilibrium and the exact answers
are the right comparison.

**The refusals.** The comparison is asked for on five configurations and granted on one. Random seating,
a shaped reward, a player count this folder carries no profile for, and a player count with no exact
answer on record are each refused with the reason.

## What N changes, and why that is stated rather than hidden

Steps 7 and 8 could each show that exactly one attribute differed between their two arms. **Player count
cannot**: it moves the accuracies, the critic's input width (9, 15, 21), the number of decisions per
episode and both refusal thresholds at once. Which is precisely why each count is checked against its own
exact answer — this is not one game with a dial turned, it is three games each validated separately.

**A larger field is also a more tightly packed one.** Neighbouring accuracies differ by 0.2000 at N=3,
0.1000 at N=5 and 0.0667 at N=7. That tightening is **not** what reverses the exact advantage, and folder
02's step 4 shows why with the solver: at a fixed player count a tighter field generally raises the
weakest player's advantage, and with the spacing frozen the advantage still falls as players are added.

## Reproducibility

The notebook ends by running the same thing twice from the same seed and printing whether the
two results are identical. **Two streams have to be reset, not one:** the environment has its own
generator, and a trained policy samples from torch's global stream.

## Where this sits

- Previous: [`../step8-reward-profile/`](../step8-reward-profile/) — Does the finding survive the reward that produced it?
- The whole investigation: [`../README.md`](../README.md)
- This is the last step of the folder. It ends on a measurement, and one of its three gates fails.
