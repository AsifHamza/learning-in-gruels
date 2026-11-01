# Step 2 — Three players, and the number 8/27

## What this step is

The central number of the whole project, computed exactly and then checked against published
results.

Step 1 gave one idea: **throw away the steps where nothing happens, and weight what can happen by
hazard.** This step applies it to three players and lands on

$$P_A = \tfrac{8}{27} \approx 0.2963, \qquad P_B = \tfrac{9}{27} \approx 0.3333, \qquad P_C = \tfrac{10}{27} \approx 0.3704$$

for accuracies 0.8, 0.6 and 0.4. It does it twice — by hand on paper, then with code that works
for any accuracies — and the code is then tested against six rows from Kilgour & Brams, *The
Truel*, Mathematics Magazine 70(5), 1997, page 317.

It also answers a question step 1 could not: is "shoot your strongest opponent" actually the
*right* strategy, or just one guess among many?

## How to run it

```bash
../.venv/bin/jupyter lab step2_truel_n3.ipynb
```

**Instant.** No seed, nothing random.

## What you should see

The mechanism, before any arithmetic:

```
hazard aimed at A: 1.0
hazard aimed at B: 0.8
hazard aimed at C: 0.0

B dies first: 0.444444  = 4/9 ? True
A dies first: 0.555556  = 5/9 ? True
```

The triple, by hand:

```
player   survival        as a fraction of 27
  A      0.2962962963    8.000000 / 27
  B      0.3333333333    9.000000 / 27
  C      0.3703703704    10.000000 / 27

P_A == 8/27 : True
P_B == 9/27 : True
P_C == 10/27: True
```

The general solver agreeing with the hand version to twelve decimal places, using only 6 states.

Then the real test — six published rows:

```
  a     b     c   |  published P_A P_B P_C  |  computed              | worst error
------------------------------------------------------------------------------------
 0.90  0.50  0.10  |  0.540 0.333 0.127     |  0.54000 0.33333 0.12667  |  0.00033
 0.90  0.50  0.30  |  0.397 0.294 0.309     |  0.39706 0.29412 0.30882  |  0.00018
 0.80  0.70  0.20  |  0.376 0.412 0.212     |  0.37647 0.41176 0.21176  |  0.00047
 0.60  0.50  0.25  |  0.314 0.370 0.316     |  0.31373 0.37037 0.31590  |  0.00037
 0.80  0.50  0.40  |  0.314 0.294 0.392     |  0.31373 0.29412 0.39216  |  0.00027
 0.80  0.60  0.40  |  0.296 0.333 0.370     |  0.29630 0.33333 0.37037  |  0.00037

worst disagreement across all six rows: 0.000471
=> every row matches to the precision they were printed at.
```

The paper prints three decimal places, so 0.0005 is the best agreement possible. The real
project's own solver reports the same worst error, 0.00047.

The equilibrium search over all 27 possible combinations of choices:

```
combinations checked: 27
equilibria found: 1

  A(0.8) -> B(0.6) | B(0.6) -> A(0.8) | C(0.4) -> A(0.8)
     survival: A=0.29630  B=0.33333  C=0.37037
```

And the test of whether everybody abstaining could be stable:

```
if everybody abstains, each gets: [0.3333, 0.3333, 0.3333]

  A(0.8): switching to shoot B(0.6) gains +0.33333
  B(0.6): switching to shoot A(0.8) gains +0.26667
  C(0.4): switching to shoot A(0.8) gains +0.06667

Everyone gains by breaking ranks, so all-abstain is not an equilibrium.
```

## What to look at

**The mechanism is visible before the arithmetic.** With A=0.8, B=0.6, C=0.4 under
shoot-the-strongest: A shoots B, B shoots A, C shoots A. **Two guns point at A, one at B, and
none at C.** Being the best shot made A the most dangerous person to leave alive; being the worst
made C invisible. Everything after that is multiplication.

**Three of the six published rows do *not* show survival-of-the-weakest.** With 0.9 / 0.5 / 0.1
the best shot simply wins, 0.540 to 0.127. So it is **not a law of the game** — the effect needs
the accuracies to be reasonably close together. If the weakest is far too weak, being ignored is
not enough to save them. The project's 0.8 / 0.6 / 0.4 sits at the end where the reversal is
complete, and that is a choice, not a coincidence.

Those reversed rows are also what makes the test strong: a solver that reproduces all six cannot
have been accidentally fitted to one memorised answer.

**Exactly one of 27 combinations is an equilibrium, and nobody abstains in it.** So 8/27 is not
"what happens under one particular guess" — it is *the* outcome, the only one nobody has a reason
to deviate from. And all-abstain is not stable: every single player gains by breaking ranks,
including the weakest, who gains +0.067.

That second fact matters to the project's history. The original proposal expected the weakest
player to survive by "firing in the air" and required agents to miss deliberately at least 5% of
the time. The criterion was withdrawn. The idea comes from a **different version** of this game
where players take turns in a *fixed* order; when the order is random, abstaining is simply a
wasted turn. Folder 01 showed this by simulation; this notebook shows it by exhaustion.

**Section 2.11 settles folder 01's loose end.** Folder 01 noticed the middle player surviving
almost exactly 1/3 under four different policies. The algebra here:

$$P_B = \frac{b+c}{(b+c)+a} \cdot \frac{b}{b+c} = \frac{b}{a+b+c}$$

The $(b+c)$ cancels. So $P_B = 1/3$ exactly when $b$ is one third of the total — that is, when
$b$ is the *mean*, which is exactly when the three accuracies are **evenly spaced**. The project
always spaces them evenly, so it always shows up. Change the spacing (try 0.8 / 0.7 / 0.2) and it
goes away.

## What it deliberately doesn't do

**No mixed strategies.** Only pure choices are enumerated — shoot this one, shoot that one, or
abstain. A player playing "shoot left 30% of the time" is not considered.

**No graph constraint**, because at N=3 there isn't one: a triangle is a complete graph. This is
also why shoot-strongest is the equilibrium here and stops being one at N=5.

**Only the `terminal` payoff** — +1 to the sole survivor, 0 to everyone else. Adding kill bonuses
would be a different game whose answers are not comparable.

**Next:** `step3-circle-n5/`, where the graph finally starts to matter, the targets stop existing
in any publication, and the project's real open problem appears.
