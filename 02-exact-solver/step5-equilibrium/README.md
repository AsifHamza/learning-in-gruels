# Step 5 — How you know it is an equilibrium, and what the equilibrium does

## What this step is

Steps 2, 3 and 4 all called something "the equilibrium" and moved on. This step stops and asks
two questions that were taken for granted:

1. **What is actually being checked, and why is checking it one state at a time enough?** The
   solver only ever tests single changes in single states. That looks too weak — a player could
   surely gain by changing its plan in several states at once. The reason it is not too weak is
   the most important idea in this folder, and it is short.
2. **What does the equilibrium do?** Not the survival numbers it produces — the *policy*. Who
   shoots whom. That is a different question from "who survives", and the two do not answer each
   other.

The second one matters because folder 03 judges agents on it. An agent can land on the right
survival numbers by following a rule that looks nothing like the equilibrium, so the survival
vector alone cannot tell a correct policy from a lucky one.

## How to run it

```bash
../.venv/bin/jupyter lab step5_equilibrium.ipynb
```

About **10 seconds**. No seed, nothing random.

## What you should see

### The criterion, and why one state at a time is enough

The test applied to a candidate is: **no player can raise its own survival probability by
changing its own action in a single state, with everyone else's plan — and its own plan
everywhere else — left alone.**

Two facts about this game make that sufficient rather than merely necessary.

**A kill strictly shrinks the alive-set, so no state can ever recur.** The state is just the set
of who is alive, and it only ever loses members. So the graph of states is a **directed acyclic
graph**, and there is no way to return to a state and exploit a plan you have already committed
to. The notebook checks this rather than asserting it: every successor of every state has exactly
one fewer player.

**A miss leaves the state unchanged, and that self-loop is already folded into the value.** When
a shot misses, play continues in the same state, so a naive reading gives an infinite regress —
the value of a state depends on itself. The recursion resolves it by conditioning on *which shot
lands next*: each player's accuracy is a hazard, the hazards are normalised, and the value is the
hazard-weighted average over the successor states. The waiting time disappears because nothing
in this game depends on how long it took.

Put together: because continuations are fixed and unreachable-again, a plan that cannot be
improved one state at a time cannot be improved by any combination of changes either. So passing
the test means the profile is **Markov perfect**, not merely a Nash equilibrium of the whole game
viewed from the opening position.

That is the thing to be able to say out loud. The rest of this notebook is consequences.

### Shoot-strongest is an equilibrium at N=3 and at no larger N

Running the criterion against `shoot_strongest`, and counting every profitable deviation. The unit is a (state, player, alternative action) triple, so a player with two better actions in one state counts twice:

```
   N   profitable deviations from shoot-strongest
   3        0
   4        3
   5        18
   6        67
   7        201
```

At N=3 there is nothing to improve, which is why steps 1 and 2 could treat the heuristic and the
equilibrium as the same object. From N=4 they are different objects, and the gap widens fast.

**The worked example, printed rather than described.** At N=5 with everyone seated in accuracy
order, look at p2 — accuracy 0.60, sitting between 0.50 and 0.70:

```
  p2 has accuracy 0.60
  its neighbours are p1 (0.50) and p3 (0.70)

  shoot-strongest fires at p3 (0.70): survival 0.08870
  firing at p1 (0.50) instead:  survival 0.13622
  gain: +0.04752
```

Shooting the weaker neighbour is worth about half again as much survival. The reason is that
firing at the strong neighbour on your right is doing the player on your *left* a favour — you
are removing the threat that would otherwise have been aimed past you — while leaving yourself
next to whoever survives. Being surrounded by weak players is worth more than removing a strong
one.

### What the equilibrium actually does

P(shoot the stronger of your two neighbours), by accuracy rank, weakest first:

```
  N=3   1.00000  1.00000  1.00000
  N=5   1.00000  0.39790  0.31072  0.37137  1.00000
  N=7   1.00000  0.47118  0.25172  0.20883  0.25328  0.53909  0.79893
```

**The shape is a U.** The two extremes fire at their stronger neighbour almost always; everyone
in the middle mostly fires at their *weaker* one. Neither `shoot_strongest` nor a coin flip
produces that shape: `shoot_strongest` is flat at 1.0 by construction, and a coin flip is flat at
0.5. A rule as simple as always shooting the left-hand neighbour does produce a U under ranked
seating, because the survivors stay in order round the circle, so the shape alone does not
identify the equilibrium; the size of the middle values does.

At N=3 the U is absent and that is correct, not a failure: with three alive every player's two
neighbours are the other two players, so "the stronger neighbour" is the strongest opponent and
the equilibrium coincides with the heuristic. Looking for a U-shape at N=3 is the wrong test.

**And targeting depends on how many are left:**

```
N=5:
  5 alive: 0.4000
  4 alive: 0.5000
  3 alive: 1.0000
  higher with fewer alive: True
  rises at every step: True

N=7:
  7 alive: 0.1429
  6 alive: 0.4517
  5 alive: 0.4000
  4 alive: 0.5000
  3 alive: 1.0000
  higher with fewer alive: True
  rises at every step: False
```

Every game ends the same way — with three alive the equilibrium fires at the stronger neighbour
with probability 1.0000, which is exactly shoot-strongest. But it does **not** get there by rising
at every step: at N=7 it dips from 0.4517 to 0.4000 partway down. So the defensible statement is
that targeting *depends on* the survivor count and ends at 1.0, not that it climbs steadily toward
it.

Either way, **a policy that ignores how many players remain cannot be optimal** — which is why
`n_alive` is in the observation the agents in folder 03 receive.

### How the decisions are weighted

These figures are shares of **decisions**, which is how folders 03 and 04 measure targeting on
trained agents, so the two can be compared directly. Two things decide how much a state counts.

The first is how likely the equilibrium is to **reach** it. States are not equally likely, and the
spread is large:

```
  reachable: 6 of 7 possible
  spread: 0.0952 to 0.3016
  never dies first under the equilibrium: [5]
```

One first-death state has probability zero: under the equilibrium there is a player nobody ever
shoots first. Averaging over states with equal weights would describe a game nobody plays.

The reach probabilities are checked against a number already trusted rather than taken on faith.
The probability of reaching the state "only player *i* is left" must equal *i*'s survival
probability, and it does — to 0.00e+00 at N=3 and 2.78e-17 at N=5 and N=7.

The second is how many **decisions** are made in a state once it is reached. A miss leaves the state
unchanged, so a state lasts until a shot lands. If the shooters' accuracies add up to $H$ and $m$
players are alive, each step ends the state with probability $H/m$, so it lasts $m/H$ steps on
average and each player is drawn in $1/m$ of them: $1/H$ decisions per player per visit. So each
decision is weighted by reach divided by $H$. Counting each state once per visit instead would give
0.31590 / 0.23098 / 0.28869 for the N=5 middle ranks instead of 0.39790 / 0.31072 / 0.37137, which
describes visits, not the decisions an agent's targeting is counted over. The notebook prints both.

### Archetti's projection, and what it is worth

There is a published rule for the *complete*-graph version of this game with four or more players
(Archetti 2012): shoot the type immediately weaker than yourself, unless you are the weakest, in
which case shoot the strongest. Projected onto a circle, where you can only reach two players, it
becomes: **shoot the strongest of your neighbours who is still weaker than you; if both are
stronger, shoot the stronger one.**

It is the only reference in this project with the equilibrium's U-shape, so it is worth measuring,
and worth being honest about. Under `ranked` seating the circle stays sorted as players die, so the
strongest neighbour still weaker than you is always the one on your left, and the weakest player's
left-hand neighbour is the strongest: the rule is the same as "always shoot left". Folder 04 measures
it under the name `next_weaker`, and its U-shape comes from the seating rather than from any reading
of strength.

```
   N   Archetti mean gap   worst single rank   shoot-strongest mean gap   ratio
   3       0.3333              1.0000             0.0000                ss exact
   4       0.1731              0.3597             0.2598                1.5x
   5       0.1164              0.2177             0.3840                3.3x
   6       0.1585              0.5037             0.3921                2.5x
   7       0.1505              0.2603             0.4967                3.3x
```

**It is closer than `shoot_strongest` — 1.5 to 3.3 times closer — but it does not converge.** The
gap at N=7 (0.1505) is barely smaller than at N=4 (0.1731), and it rises in between. It is an
approximation that stays an approximation.

**It undershoots the middle**, which is where the interesting behaviour is. At N=5 the middle
three ranks sit at 0.2013 / 0.0930 / 0.2038 against the equilibrium's 0.3979 / 0.3107 / 0.3714 —
right about the direction, 0.30 to 0.55 of the magnitude.

**Applied at N=3, the four-player rule is qualitatively wrong**, giving 1.0000 / 0.0000 / 1.0000
where the equilibrium is 1.0 everywhere: it misses the middle player by a full 1.0. That is a limit
of the projection, not of Archetti, whose three-player rule is different (everyone shoots the
strongest opponent) and coincides with the equilibrium.

Its value is therefore narrow and real: above N=8 no exact answer is on record, so there is no
targeting reference at all, and this is the only candidate with the right shape.

## What to watch for

**The deviation counts are not a quality score.** 201 deviations at N=7 does not mean
`shoot_strongest` is 201 times worse than at N=4 — it means there are more states and more
players in them. What the counts establish is the qualitative fact that the number is zero at
N=3 and non-zero everywhere after.

**Targeting weighted by decisions, by visits and uniformly over states are three different
numbers**, and comparing one against another produces a wrong answer that looks reasonable. Every
figure in this notebook is weighted by decisions, including Archetti's and `shoot_strongest`'s,
each under its **own** play — because Archetti's play leads to different states, and different
hazards, than the equilibrium's does.

**"The equilibrium" is singular here, and the notebook checks that.** Where a state has several
equilibria a choice has to be made, and a policy read off a chosen equilibrium is partly an
artefact of the choice. Step 4 established that no state in the solvable range has more than one,
so that concern does not arise — but it is checked, not assumed.

## What this step feeds

Folder 03 measures exactly this quantity on its trained agents, calling it the *targeting rate*,
and reads it against the three shapes this notebook makes concrete: a U with survivor-count
dependence means the equilibrium; flat and high means the heuristic; flat near 0.5 means the
policy is discriminating on nothing.

That completes deliverable 02. Folder `03-agent-architecture/` trains neural network agents that
have to rediscover these numbers without being told them.
