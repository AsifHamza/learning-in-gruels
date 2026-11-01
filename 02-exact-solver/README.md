# Deliverable 02 — The Exact Solver

**What this deliverable is:** the analytical answer. It computes survival probabilities *exactly*,
by arithmetic, with no simulation and no randomness anywhere.

This is where the **targets** come from. Folder 03 trains neural network agents, and the only way
to know whether they learned the right thing is to already know the right thing. That is this
folder's whole job.

It needs no libraries at all. Not numpy, not torch — the solver is plain Python. The hardest
mathematics in the project has the fewest dependencies, which is worth noticing.

---

## Why bother, when folder 01 already measured it?

Folder 01 played 100,000 episodes to estimate that the weakest player survives 0.3705 of the time.
The true value is 0.37037. It was close, but:

- it is never exact — there is always sampling error left over;
- the error shrinks like $1/\sqrt{M}$, so **four times** the episodes to halve it;
- and it can only tell you what *did* happen under one policy, never whether that policy was the
  right one.

This folder gets 0.370370370370 in four thousandths of a second, and can additionally prove that no
player would rather do something else. Simulation cannot do that second part at all.

---

## What to expect

**N=3**, accuracies 0.8 / 0.6 / 0.4:

| Player | Survival | Fraction |
|---|---|---|
| 0.8 | 0.296296… | 8/27 |
| 0.6 | 0.333333… | 9/27 |
| 0.4 | **0.370370…** | 10/27 |

**N=5 circle, ranked seating** — the numbers folder 03 will be held to. They appear in no
publication; this folder produces them:

| Accuracy | Survival | Rank by survival |
|---|---|---|
| 0.8 (strongest) | 0.19804 | 3rd |
| 0.7 | 0.16194 | **5th — worst** |
| 0.6 | 0.17970 | 4th |
| 0.5 | 0.21073 | 2nd |
| 0.4 (weakest) | **0.24959** | 1st — best |

They sum to exactly 1, as they must, since exactly one player survives.

---

## The steps

Read them in order. Every step folder has its own README with the exact commands and the exact
output to expect.

| Folder | What it teaches | Runtime |
|---|---|---|
| [`step1-duel/`](step1-duel/) | Two players. Derives $W(p,q) = p/(p+q)$ and the cancellation that makes the rest easy | instant |
| [`step2-truel-n3/`](step2-truel-n3/) | Three players. 8/27 by hand, then in code, then checked against six published rows. Proves shoot-strongest is the unique equilibrium | instant |
| [`step3-circle-n5/`](step3-circle-n5/) | Five players, where the graph matters. Produces the N=5 targets, and states the open problem the project has not solved | ~3 s |
| [`step4-how-far/`](step4-how-far/) | Runs the solver upward until it fails. Reaches N=8, refuses at N=9 because a state has no pure equilibrium, and shows that state has a mixed one. Also: the advantage to the weakest peaks at N=6 and reverses | ~5 s |
| [`step5-equilibrium/`](step5-equilibrium/) | Why a one-shot deviation test is enough, and what the equilibrium's **policy** is — the U-shaped targeting rule folder 03 measures its agents against | ~10 s |

No seeds anywhere. Nothing here is random.

---

## Running any of it

```bash
cd 02-exact-solver
python3 -m venv .venv                            # first time only
./.venv/bin/pip install -r requirements.txt      # first time only
./.venv/bin/jupyter lab
```

Or from inside a step folder:

```bash
cd step2-truel-n3
../.venv/bin/jupyter lab step2_truel_n3.ipynb
```

Notebooks are saved with their outputs, so they can be read without running. Each step folder holds
two files with the same content — read the **`.ipynb`**, ignore the `.py` unless you want to diff
it in git.

---

## The one idea the whole folder rests on

From step 1. Two players, accuracies $p$ and $q$, one picked at random each step:

$$W = \tfrac12\big[p + (1-p)W\big] + \tfrac12\big[(1-q)W\big] \qquad\Longrightarrow\qquad W = \frac{p}{p+q}$$

**The coin flip cancels and the waiting time cancels.** Only the ratio of accuracies survives.
Whether the duel lasts 2 steps or 200 makes no difference to who wins.

It cancels because the game has **no memory**: when a shot misses, the situation is exactly what it
was before, so its value is the same number. Equivalently — and this is the version that
generalises — throw away the steps where nothing happens, and weight each thing that *can* happen
by how fast it happens. Those weights are the shooter's accuracies, and they are called **hazards**.

Every solve in this folder is that idea applied repeatedly:

$$V_i(S) = \sum_{t \in S} \frac{h_t}{\sum_u h_u} \; V_i(S \setminus \{t\})$$

where $S$ is who is still alive and $h_t$ is the total hazard aimed at player $t$. Missed shots do
not appear. Neither does the $1/m$ for whose turn it is, because every term has it.

---

## Results in here that are more interesting than the numbers

**Survival-of-the-weakest is not a law.** Of the six published N=3 rows, **three do not show it**.
With accuracies 0.9 / 0.5 / 0.1 the best shot simply wins, 0.540 to 0.127. The effect needs the
accuracies reasonably close together — if the weakest is far too weak, being ignored does not save
them. The project's 0.8 / 0.6 / 0.4 sits where the reversal is complete, and that is a choice.

**Nobody abstains in equilibrium.** At N=3 there are 27 possible combinations of choices and exactly
**one** is an equilibrium: everybody shoots their strongest opponent, and nobody deliberately
misses. All-abstain is not an equilibrium — every player gains by breaking ranks, including the
weakest. This matters historically: the project's original proposal expected the weakest player to
survive by "firing in the air" and required agents to miss on purpose at least 5% of the time. That
criterion was withdrawn. The idea comes from a different version of the game where players take
turns in a **fixed** order; when the order is random, abstaining is a wasted turn.

**At N=5, shoot-strongest stops being the equilibrium.** There are **18** profitable deviations from
it, and it produces a completely different profile (0.410 for the weakest instead of 0.250). At N=3
the two coincide because the triangle is a complete graph. So the N=5 comparison must be against the
solved equilibrium, never the heuristic — a distinction easy to miss precisely because N=3 does not
force it.

**The middle player's 1/3 is neither coincidence nor law.** Folder 01 noticed the middle accuracy
surviving almost exactly 1/3 under four different policies. The algebra: $P_B = b/(a+b+c)$, which is
1/3 exactly when $b$ is the mean of the three — i.e. when the accuracies are **evenly spaced**. The
project always spaces them evenly, so it always shows up.

---

## The open problem, stated honestly

Everything at N=5 above uses **ranked** seating — accuracies placed in order around the circle. Under
ranked seating the equilibrium is unique in all 26 states and survival is **non-monotone**: the
weakest does best, but the **second-strongest does worst of anyone**.

Averaged over all 120 possible seatings, that non-monotone shape **disappears** and survival becomes
monotone in accuracy. So the striking result is caused by *seating*, not by the graph. The real
project initially attributed it to the graph constraint and had to withdraw the claim.

That leaves a problem this folder does **not** solve. The environment reshuffles seating every
episode, so the agents face random seating — but each of the 120 solves assumes the players *know*
the arrangement. Our agents see only `[own accuracy, left neighbour, right neighbour, fraction
alive]` and cannot tell which seating they are in. Their game is one of **incomplete information**,
and the equilibrium of such a game is not in general the average of the equilibria of the games it
might be.

So:

- **ranked seating** has a defensible target, because nothing is hidden;
- **random seating** does not yet, because nobody has solved the incomplete-information version.

This is the project's main open question. Section 3.9 of step 3 lays it out; it does not resolve
it.

---

## Where this method stops, and why

The reason folders 03 and 04 exist. States grow as $2^N$, and each state needs every combination of
choices checked — up to $3^N$ of them.

| N | States | Combinations in the full state | Rough total checks | Solve time measured here |
|---|---|---|---|---|
| 3 | 4 | 27 | 108 | instant |
| 5 | 26 | 243 | 6,318 | ~0.004 s |
| 6 | 57 | 729 | 41,553 | ~0.02 s |
| 7 | 120 | 2,187 | 262,440 | ~0.1 s |
| 8 | 247 | 6,561 | 1,620,567 | ~0.4 s, in step 4 |
| 9 | 502 | 19,683 | 9,880,866 | **refused** in step 4 — no pure equilibrium |
| 10 | 1,013 | 59,049 | 59,816,637 | not reached |
| 15 | 32,752 | 14,348,907 | 469,955,402,064 | refused at a state with 8 alive |

Roughly a factor of 5 per player added, and at N=15 the count is very large —
$4.7 \times 10^{11}$ checks.

**But cost is not what stops this method, and it is worth being exact about that.** Carry the
same factor of 5 forward from the ~0.1 s measured at N=7 and N=10 lands near 10 seconds,
which is nobody's definition of impossible. What actually happens is that the method runs
out of *pure equilibria* before it runs out of time: at **N=9 there is a state with no pure
Nash equilibrium at all**, so backward induction over pure strategies reaches a point where there
is nothing to select and it stops. The same happens at 10 and 15 players.

That state does have a **mixed** equilibrium, in which two players randomise between a shot and
holding fire (step 4). So the limit is on this method, not on the game: a solver over mixed
strategies could go further, and this project does not build one. What matters for the project is
that above N=8 there is no exact answer on record, so the agents there can be described but not
validated.

---

## What this folder deliberately does not do

- **No simulation.** Not a single random number.
- **No learning.** Folder 03.
- **No shaped rewards.** The solver assumes the `terminal` payoff — +1 to the sole survivor, 0 to
  everyone else — because that is the game the targets are defined for.
- **No mixed strategies.** Only pure choices are enumerated. A player playing "shoot left 30% of the
  time" is not considered.
