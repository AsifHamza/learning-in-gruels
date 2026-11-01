# Step 1 — The duel, and the cancellation that makes everything easy

## What this step is

Two players. It is the smallest possible version of the game, and it contains the one idea that
makes the whole folder short.

Folder 01 measured survival by playing 100,000 episodes. That works, but it is never exact —
there is always about 0.0015 of wobble left, and halving it needs four times the episodes. This
folder computes the answer exactly instead. No episodes, no randomness, no error bars, and no
libraries at all.

The result: two players with accuracies $p$ and $q$, one picked at random each step,

$$W(p,q) = \frac{p}{p+q}$$

**The coin flip cancels. The waiting time cancels.** Only the *ratio* of the two accuracies
survives. Whether the duel takes 2 steps or 200, and whether the players shoot often or rarely,
makes no difference to who wins — only their relative deadliness does.

## How to run it

```bash
../.venv/bin/jupyter lab step1_duel.ipynb
```

**Instant.** No seed, because nothing is random.

## What you should see

The formula, evaluated:

```
p=0.8 vs q=0.4 -> 0.6666666666666666
p=0.4 vs q=0.8 -> 0.3333333333333333
p=0.6 vs q=0.4 -> 0.6
p=0.5 vs q=0.5 -> 0.5

adds to 1: 1.0
```

A second, independent way of getting the same answer — solving the equation by repeatedly
plugging the guess back in:

```
accuracies      formula      iteration     difference
  0.8 vs 0.4     0.6666666667   0.6666666667   0.00e+00
  0.6 vs 0.4     0.6000000000   0.6000000000   0.00e+00
  0.9 vs 0.1     0.9000000000   0.9000000000   1.11e-16

Same answer to ten decimal places. The algebra is right.
```

And the case folder 03 will lean on:

```
weak player (0.4) survives: 0.3333333333333333 = 1/3 ? True
strong player (0.8) survives: 0.6666666666666666 = 2/3 ? True
```

## What to look at

**Why the self-reference is legal.** The equation is

$$W = \tfrac12\big[p + (1-p)W\big] + \tfrac12\big[(1-q)W\big]$$

with $W$ on both sides. That is allowed because the game has **no memory** — when a shot misses,
the situation is exactly what it was before: same two players, same accuracies, same rules. So
the value of being there is the same number. This collapse is why the solver in step 2 fits on a
page instead of summing an infinite series.

**Section 1.5 is the version that generalises.** Forget the timing entirely and ask only: who
lands the first hit? Per step, $p$ hits with probability $\tfrac12 p$ and $q$ with $\tfrac12 q$.
"Nothing happens" just means try again, and it favours nobody. So conditional on *someone* being
hit, it was $p$ with probability $\tfrac12 p / (\tfrac12 p + \tfrac12 q) = p/(p+q)$. Same answer,
no algebra.

The recipe, and everything in step 2 is this recipe applied repeatedly:

> **Throw away the steps where nothing happens. Weight each thing that can happen by how fast it
> happens.**

Those weights — the shooter's accuracy — are called **hazards**.

**A duel has no survival-of-the-weakest.** The better shot simply wins more often, in exact
proportion to how much better it is. The interesting effect needs at least **three** players,
because it depends on opponents choosing *each other* as targets. It is a fact about targeting,
not about shooting.

**Remember $W(0.4, 0.8) = 1/3$.** Folder 03 will use the two-player duel as its fast check that
training works, because it converges in seconds and the right answer is a simple fraction. The
three-player game needs roughly 500,000 training steps before even the *sign* of the effect is
trustworthy, so it is useless as a quick test.

## What it deliberately doesn't do

**No three-player case** — that is step 2.

**No graph.** With two players there is only one edge, so adjacency cannot restrict anything.

**No abstention.** Both players always fire here. Step 2 adds abstaining and shows the
equilibrium never uses it.

**Next:** `step2-truel-n3/` applies the hazard idea to three players, lands exactly on 8/27,
9/27, 10/27, and checks itself against six rows from a 1997 paper.
