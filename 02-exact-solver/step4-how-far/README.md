# Step 4 — How far the exact answers reach, and what they do on the way

## What this step is

Steps 1 to 3 solved three games: N=2, N=3, N=5. This step stops solving particular games and
asks how far the method goes at all — by running it upward until it fails, and watching what
it says about the players as the field grows.

Three questions get separated here, because they are usually run together and only one of them
has the answer people expect:

1. **Where does the exact method stop?** Demonstrated, not estimated.
2. **Why does it stop?** Not for the reason the cost arithmetic suggests.
3. **What do the answers do as N rises?** The survival-of-the-weakest effect does not keep
   growing. It peaks and then reverses.

Nothing here is new code. The solver is step 3's, copied forward unchanged, which is the point:
the boundary is a property of the game, not of a particular implementation.

## How to run it

```bash
../.venv/bin/jupyter lab step4_how_far.ipynb
```

About **5 seconds**, most of it the N=8 solve and the refused N=9. No seed, nothing random — the
same reason as every other step in this folder: there are no episodes here, only enumeration.

## What you should see

### The walk upward, and where it ends

Each player count is solved from scratch, smallest state first. It gets to **N=8** and then
refuses at **N=9**:

```
  N=3:    4 states, 0 with more than one equilibrium, 0.00 seconds
  N=4:   11 states, 0 with more than one equilibrium, 0.00 seconds
  N=5:   26 states, 0 with more than one equilibrium, 0.01 seconds
  N=6:   57 states, 0 with more than one equilibrium, 0.02 seconds
  N=7:  120 states, 0 with more than one equilibrium, 0.10 seconds
  N=8:  247 states, 0 with more than one equilibrium, 0.45 seconds
  N=9: REFUSED after 1.69 seconds
        no equilibrium at state (0, 1, 2, 4, 5, 6, 7, 8)
```

The seconds are wall-clock on one machine and will differ on yours. The state counts and the
refusal are fixed.

**The wall is not cost.** Counting states and strategy combinations, as step 3 does, shows the
work growing steeply enough that running out of time looks like the obvious limit. The timings
here say otherwise: each extra player costs roughly **four to five times** the last, so N=9 would
have been about two seconds and N=10 under ten. Both are affordable, and the solver never got
the chance to be slow.

What actually happens is that at N=9 a state has **no pure equilibrium at all** — no
combination of definite choices that every player is content with. The notebook confirms this by
checking all **6,561** pure combinations available in that state and finding **zero** stable ones:

```
  combinations of choices in this state: 6561
  of those, stable (nobody wants to deviate): 0
```

**But the state does have a mixed equilibrium.** If two players randomise, p7 shooting p6 with
probability 0.1303 and p8 shooting p7 with probability 0.4096, each holding fire otherwise, while
the other six each play one action, then no player gains by switching to any single action:

```
state (0, 1, 2, 4, 5, 6, 7, 8)
  p7 shoots p6 with probability 0.1303 and holds fire otherwise
  p8 shoots p7 with probability 0.4096 and holds fire otherwise
  largest gain from switching to any single action: 0.0e+00
  an equilibrium: True
```

So the limit is on the method, backward induction over **pure** strategies, and not on the game.
The notebook verifies this one mixed equilibrium; which players mix was found by a search outside
the notebook, and whether every larger state has a mixed equilibrium is not checked. A solver over
mixed strategies could go further, and this project does not build one. Above N=8 there is no exact
answer on record, which is what makes folder 04 a description rather than a validation.

### The trend, which reverses

Survival of the weakest player, as a multiple of an even `1/N` share:

```
  N   weakest x uniform   weakest / strongest   shoot-strongest x uniform
  3         1.11                 1.25                    1.11
  4         1.18                 1.25                    1.58
  5         1.25                 1.26                    2.05
  6         1.31                 1.66                    2.52
  7         1.07                 1.23                    3.00
  8         0.77                 0.96                    3.47
```

**The advantage peaks at N=6 and then reverses.** At N=7 the weakest player is no longer even
the best-off player in the game. At N=8 it drops *below* an even share and ends up worse off
than the strongest player.

The notebook then draws the weakest and strongest players' exact survival, each as a multiple of
an even share, from three to eight players. This is the chart the capstone report uses for the
same result.

The third column is the same measurement under `shoot_strongest`, and it climbs at every single
step. So **a ratio that grows with player count is that heuristic's signature, not evidence of
good play** — which is why folder 03 gates on reproducing the whole profile rather than on
beating anyone.

**Is the reversal caused by the field getting tighter?** Accuracies are always spread evenly over
[0.4, 0.8], so the gap between neighbouring players shrinks from 0.2000 at N=3 to 0.0571 at N=8.
A natural guess is that being the weakest matters less when everyone is nearly equally good, and
that this is what reverses the advantage. The notebook tests that guess with the solver, two ways,
each changing one thing:

| Test | What is held fixed | What it finds |
|---|---|---|
| 1 | the player count, with the middle accuracy at 0.6 | a tighter field **raises** the weakest's advantage: from 1.190 to 1.444 at N=5, and at every count tested the tightest spacing beats the loosest |
| 2 | the spacing between players | the advantage **still** ends below its peak as players are added, at every spacing tested |

So the guess is the wrong way round. Tighter packing generally helps the weakest, and the reversal
happens without it. One exception to "generally": at N=8, going from a gap of 0.07 to the standard
0.0571 lowers the advantage slightly (0.819 to 0.765) before tighter spacings raise it again.

Two honest limits. Freezing the spacing in test 2 means a larger field spans a *wider* range of
accuracies, so that test moves something too. And what *does* drive the reversal is not settled
here. Some of the games in test 2 have no pure equilibrium at all — the same wall described above,
met earlier when the accuracies are spread wider.

### The one state where somebody holds fire

Across every state of every game from N=3 to N=8 — 465 states in total — there is **exactly
one** where a player in equilibrium chooses not to shoot. It is the opening state at N=8, and
the two players who hold fire are the second- and third-strongest, sitting next to each other:

```
   player  accuracy   action        survival
     p0     0.4000    shoot p7      0.09565
     p1     0.4571    shoot p0      0.15536
     p2     0.5143    shoot p1      0.13454
     p3     0.5714    shoot p2      0.11978
     p4     0.6286    shoot p3      0.13643
     p5     0.6857    ABSTAIN       0.13745
     p6     0.7429    ABSTAIN       0.12147
     p7     0.8000    shoot p0      0.09932
```

Two things make this worth stating carefully.

**It is not an artefact of choosing between equilibria.** Step 3 introduced a tie-break for
states with more than one equilibrium — prefer the one with the most shooting. The notebook
counts how often that tie-break was needed across all six games and the answer is **never**:
every state in the solvable range has exactly one equilibrium. So this is not the solver
picking a quiet equilibrium over a loud one. There was no choice to make.

**The margins are thin, and are printed rather than described.** Holding fire beats p5's best
alternative by 0.000938 and p6's by 0.002186. Real, well clear of the 1e-12 tolerance, but
small — so this is reported as what it is, a narrow preference in a single state, not a
dramatic result.

**The weakest player never holds fire, at any N from 3 to 8.** In this same state it is the
player with the *most* to lose by doing so: abstaining costs it 0.008218, the largest such
margin of anyone. The classical prediction that the weakest player should deliberately miss
belongs to the fixed-order game (Kilgour 1975); this variant is a different game, and here
holding fire is a tool of the strong.

**Larger fields fail the same way (section 4.7).** The same solver, run on 9, 10 and 15 players,
refuses every time, and every time at a state with eight players alive:

```
  N=9: refused after 1.8 seconds, at a state with 8 players alive, (0, 1, 2, 4, 5, 6, 7, 8)
  N=10: refused after 4.4 seconds, at a state with 8 players alive, (0, 1, 2, 3, 4, 5, 7, 9)
  N=15: refused after 183.0 seconds, at a state with 8 players alive, (0, 1, 2, 3, 4, 5, 11, 12)
```

So nine players is not special. With accuracies spread evenly over 0.4 to 0.8, a state with eight
players alive can have no pure stationary equilibrium, and the solver has to solve every state with
eight alive before it can solve a larger game, so none of the larger games has an answer of this
kind. With other spacings the first failure can come earlier; that was not explored. The fifteen-player run solves all 16,368 states with two to seven players
alive before it reaches one it cannot solve, which is why it takes about three minutes rather than
the hours a full solve would need. The timings are wall-clock and will differ by machine.

## What to watch for

**The refusal is the result, so do not treat the exception as a failure.** The notebook catches
it deliberately and prints the state it names. A run that got to N=9 without raising would mean
something had changed in the solver.

**Two of the three columns in the trend table are different quantities.** The equilibrium
column and the heuristic column are two different games, not two estimates of one thing. They
are printed side by side to make exactly that point, and reading a single trend across them
would be an artefact of the join.

**The timings will not match to the decimal.** They are wall-clock measurements on whatever
machine runs the notebook. The ratio between consecutive rows is the durable part, not the
absolute figures.

## What this step feeds

Folder 03 trains agents at N=3, N=5 and N=7, and each of those has an exact answer here to be
held against. Above N=8 there is nothing to be held against — which is what folder 04 is for,
and is now a demonstrated claim rather than an assumed one.

[`../step5-equilibrium/`](../step5-equilibrium/) stays at the player counts this step reached and
asks a different question: how you know the thing the solver returned really is an equilibrium,
and what that equilibrium actually does.
