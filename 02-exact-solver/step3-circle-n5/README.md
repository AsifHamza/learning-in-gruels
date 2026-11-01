# Step 3 — Five players on a circle, where the graph finally matters

## What this step is

The N=5 targets — the numbers the trained agents in folder 03 will be held to.

At N=3 the circle is a triangle, so everyone can shoot everyone and the graph does nothing. At
N=5 that stops being true: you have four opponents and can only reach **two** of them.

These numbers are worth being careful about for one reason: **they are not in any paper.** At
N=3 we could check against Kilgour & Brams. Here there is nothing to check against, so the only
protection is that the same code got all six of their rows right in step 2.

The step also produces two results more interesting than the numbers, and one genuine open
problem the project has not resolved.

## How to run it

```bash
../.venv/bin/jupyter lab step3_circle_n5.ipynb
```

About **3 seconds**, most of it the 120-seating average. No seed, nothing random.

## What you should see

The state count:

```
states needing a decision: 26
  2 players alive: 10 states
  3 players alive: 10 states
  4 players alive: 5 states
  5 players alive: 1 states
```

Contraction changing who can shoot whom:

```
full circle:            after p1 dies:
  p0 can shoot [4, 1]     p0 can shoot [4, 2]
  p1 can shoot [0, 2]     p2 can shoot [0, 3]

p0 and p2 are now neighbours. They were not before.
```

The generic circle solver reproducing N=3 exactly, which it must before anything else is
trusted:

```
position  accuracy   survival       exact          match
   p0      0.4      0.3703703704  0.3703703704  True
   p1      0.6      0.3333333333  0.3333333333  True
   p2      0.8      0.2962962963  0.2962962963  True
```

**The N=5 targets**, ranked seating:

```
position  accuracy   survival
   p0      0.4      0.24959
   p1      0.5      0.21073
   p2      0.6      0.17970
   p3      0.7      0.16194
   p4      0.8      0.19804

sums to: 1.0
states solved: 26
states with more than one equilibrium: 0
=> the equilibrium is unique everywhere, so no choice had to be made.
```

Sorted by survival:

```
accuracy   survival     rank by survival
  0.8      0.19804      3 of 5
  0.7      0.16194      5 of 5
  0.6      0.17970      4 of 5
  0.5      0.21073      2 of 5
  0.4      0.24959      1 of 5
```

The heuristic from folder 01 compared with the equilibrium:

```
accuracy   equilibrium   shoot-strongest   difference
  0.4       0.24959       0.40959         +0.16000
  0.5       0.21073       0.31369         +0.10296
  0.6       0.17970       0.08870         -0.09101
  0.7       0.16194       0.07695         -0.08499
  0.8       0.19804       0.11107         -0.08697

profitable deviations from shoot-strongest, across all states: 18
=> shoot-strongest is NOT an equilibrium at N=5.
```

Averaged over all 120 seatings:

```
accuracy   ranked seating   averaged over seatings
  0.4       0.24959          0.22375
  0.5       0.21073          0.21356
  0.6       0.17970          0.19765
  0.7       0.16194          0.18395
  0.8       0.19804          0.18110
```

And where the whole method dies:

```
     N     states    combinations per full state    rough total checks
     3          4                        27                   108
     5         26                       243                 6,318
    10      1,013                    59,049            59,816,637
    15     32,752                14,348,907       469,955,402,064

one N=5 solve: 0.004 seconds
one N=6 solve: 0.020 seconds
one N=7 solve: 0.095 seconds
```

Those are wall-clock times on one machine, so your figures will differ a little. The durable
part is the ratio between them: roughly five times per extra player.

## What to look at

**Survival is not monotone in accuracy.** The weakest does best, as at N=3. But the strongest is
only **third**, and the **second-strongest does worst of everybody** — 0.16194, below even the
0.6 player. That is not a clean reversal of the ordering; it is a different shape entirely.

The reason is the graph. On a circle you can only be shot by your two neighbours, so what matters
is not how good you are but **who you are sitting next to**. Under ranked seating the
second-strongest sits directly beside the strongest, which is the worst seat at the table.

**Shoot-strongest stops being the equilibrium.** At N=3 it was the unique one. Here there are
**18** profitable deviations from it, and it produces a wildly different profile — it gives the
weakest player 0.410 instead of 0.250. So at N=5 the agents must be compared against the
*equilibrium* column, never the heuristic. That is easy to miss precisely because N=3 does not
force it.

**Then section 3.9 takes the non-monotone result away again.** Averaged over all 120 seatings the
profile becomes **monotone** — weakest best, strongest worst, in order. So the striking result is
caused by *seating*, not by the graph constraint.

**And here is the open problem.** The averaged column cannot simply be used as the target either.
Each of those 120 solves assumes the players **know** the arrangement — they are complete-
information games. Our agents see only `[own accuracy, left neighbour, right neighbour, fraction
alive]`, so they cannot tell which of the 120 seatings they are in. Their game is one of
**incomplete information**, and the equilibrium of such a game is not in general the average of
the equilibria of the games it might be. So:

- **ranked seating** has a target we can defend, because nothing is hidden;
- **random seating** does not yet, because nobody has solved the incomplete-information version.

This is the project's main open question. This notebook makes it clear; it does not resolve it.

**The cost table shows how steeply the work grows.** Roughly a factor of 5 per player added.
N=5 is instant and N=15 needs about $4.7 \times 10^{11}$ checks.

Steep growth is not the same as a boundary, though, and this table does not locate one. Step 4
locates it by running the solver upward until it fails: the exact answers reach **N=8**, and N=9
fails because a state has **no pure equilibrium at all** (it does have a mixed one). Cost would
have allowed another player or two. That distinction is why folders 03 and 04 exist — not because
the sums get big, but because past N=8 there is no exact answer on record to compare an agent
against.

## What it deliberately doesn't do

**No non-shooting equilibrium selection.** Where a state has several equilibria the code takes the
one with the most shooting. At N=5 it makes no difference — every state has exactly one — but the
real solver can also pick the abstaining one deliberately, to measure how much the answer depends
on that choice.

**No `complete` topology at N>3.** A complete graph would be one control for whether the
non-monotonicity is about seating rather than the graph. This notebook makes the point by averaging
over seatings instead.

**No mixed strategies, and no incomplete-information solve** — the second of which is the open
problem above.

**Next:** [`../step4-how-far/`](../step4-how-far/), which takes this same solver and runs it
upward until it fails — settling how far the exact answers reach, and why they stop.
