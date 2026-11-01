# Step 4 — The four baseline policies, and how to measure honestly

## What this step is

Something to compare against, and the statistics to make the comparison mean anything.

Before you can claim a trained neural network is good, you have to know what **simple** looks
like. This step builds four policies that contain no learning at all, measures them properly,
and shows how to tell a real difference from noise.

The baselines are what every trained agent is measured against. The `random` baseline matters most:
a policy whose survival numbers match it has learned nothing, however sophisticated the architecture.
That is what baselines are for — not "is my number big" but "is my number different from doing
nothing".

## How to run it

```bash
../.venv/bin/jupyter lab step4_heuristics.ipynb
```

About **35 seconds** — the slowest notebook in this folder. Four policies × 100,000 episodes,
plus three focal-agent runs of 30,000 each.

## What you should see

The four policies, side by side:

```
policy              p=0.4    p=0.6    p=0.8
---------------------------------------------
shoot_strongest     0.3705  0.3324  0.2971
shoot_weakest       0.0723  0.3326  0.5950
random              0.2213  0.3335  0.4452
always_miss         0.3333  0.3333  0.3333

policy            stagnation   deliberate misses   hard cap
------------------------------------------------------------
shoot_strongest     0.0000       0.0000            0.0000
shoot_weakest       0.0000       0.0000            0.0000
random              0.0000       0.3340            0.0000
always_miss         1.0000       1.0000            0.0000
```

`shoot_strongest` against the exact answer:

```
accuracy   exact     measured   diff       s.e.      diff/s.e.
  0.4      0.37037   0.37053   +0.00016   0.00153   +0.10
  0.6      0.33333   0.33236   -0.00097   0.00149   -0.65
  0.8      0.29630   0.29711   +0.00081   0.00144   +0.56

worst disagreement: 0.65 standard errors
=> agrees with the exact answer.
```

How the error shrinks with episode count:

```
  1,000 episodes: weakest = 0.3840, off by 0.0136, predicted s.e. 0.0153
 10,000 episodes: weakest = 0.3756, off by 0.0052, predicted s.e. 0.0048
100,000 episodes: weakest = 0.3705, off by 0.0002, predicted s.e. 0.0015
```

The focal-agent comparison — one deviator against `shoot_strongest` opponents:

```
focal plays          p=0.4   p=0.6   p=0.8
--------------------------------------------------
shoot_strongest      0.3704  0.3347  0.2953
shoot_weakest        0.3478  0.2779  0.2419
always_miss          0.0000  0.0000  0.0000
```

`always_miss` never fires at all, which nobody proposes. The classic "fire in the air" is to abstain while all three are alive and shoot once it is a duel, so the notebook also tests that, over 300,000 episodes because the effect on the weakest is small:

```
focal plays          p=0.4   p=0.6   p=0.8
--------------------------------------------------
shoot_strongest      0.3723  0.3326  0.2954
fire_in_air          0.3608  0.2009  0.0000
difference          -0.0115 -0.1316 -0.2954
s.e. of one cell     0.0015  0.0015  0.0014
```

Plus one figure: bar chart of all four policies, and an error-bar plot against the exact values.

## What to look at

**`shoot_weakest` does not merely reverse the profile — it destroys the weak player.** The 0.4
player drops from 0.3705 to **0.0723**, a factor of five. So being the weakest is either the
best or nearly the worst seat in the game, depending entirely on what everyone else has decided
to do. Survival-of-the-weakest is a fact about *targeting*, not about the rules.

**The middle player gets 1/3 under every single policy.** Read down the `p=0.6` column: 0.3324,
0.3326, 0.3335, 0.3333. Four completely different policies. Is that a general law or a
coincidence of these particular numbers? Folder 02, step 2 answers it — it turns out to be
$b/(a+b+c)$, which is 1/3 exactly when the accuracies are evenly spaced.

**No deviation beats `shoot_strongest` at any accuracy level.** In the focal table, row 1 is
above both other rows everywhere. So if everybody shoots their strongest neighbour, no single
player can do better by changing their mind. That is what makes it an **equilibrium**, and it is
why 8/27, 9/27, 10/27 is the *right* number to hold agents to rather than just one arbitrary
policy's number.

**A lone abstainer survives 0.0000 of the time.** Not "less often" — essentially never. The
other two keep firing, so an abstainer is just a target that does not shoot back.

That number matters to the project's history. The original thesis proposal expected the weakest
player to survive by *deliberately missing* — the old "fire in the air" idea — and set a success
criterion that agents must miss on purpose at least 5% of the time. The criterion was withdrawn.
The two tables are that withdrawal, measured. Never firing pays zero. Firing in the air only while all three are alive costs every seat: the weakest loses 0.0115, more than five standard errors of the difference between two cells, because the other two were already shooting each other and nobody was shooting it, and the strongest is killed every time, because both opponents fire at it while it holds fire. Survival-of-the-weakest happens anyway, through targeting alone.

**Two rules about counting that are easy to get wrong.** Count survival **by accuracy**, never
by position, because seating is reshuffled every episode. And add up **rewards**, not wins,
because a stuck episode has no winner — three players get 1/3 each — and counting winners would
silently discard those episodes so your probabilities would not add to 1.

**The arithmetic of error bars.** Standard error is $\sqrt{p(1-p)/M}$ and shrinks like
$1/\sqrt{M}$, so **four times** the episodes to halve it. At 100,000 episodes it is about
0.0015, which means differences under 0.005 are noise. At 10,000 episodes the wobble is about
0.005 — worth remembering when reading any result measured at that size.

## What it deliberately doesn't do

**No exhaustive equilibrium check.** The focal table tests three candidate deviations. There are
many more. Folder 02 checks *every* possible strategy exactly, which simulation cannot do.

**No learning agent to compare against yet.** That is folder 03, and it will reuse the focal
protocol from section 4.9.

**Only the plain game, not the PettingZoo wrapper**, because it is faster and nothing here needs
the framework.

**Next:** `step5-render/` draws the game, so contraction stops being something you take on
trust from a table of numbers.
