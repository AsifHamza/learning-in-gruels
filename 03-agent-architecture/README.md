# Deliverable 03 — The Agent Architecture

**This is the fast stab at an agent, and it is the core of the capstone.** Folders 01 and 02 built the
game and the answer key. This folder builds something that *learns* to play, then interrogates what it
learned.

The question it exists to answer:

> **What does a trained agent learn, as a function of game variant, reward profile and player count?**

Note what that is **not**. It is not *"does the agent beat a heuristic"*. A comparison against a
heuristic is one descriptive column, never a verdict, and the reason is in
[the ratio warning](#a-large-gap-against-a-heuristic-is-not-a-result) below.

**What the agent sees.** Four numbers — its own accuracy, its two neighbours' accuracies, and how much of
the field is left. Nothing further round the circle.

---

## What this folder answers to

**The brief.** Gruels generalise truels to a graph; exact strategies are computable only in small
cases; and the project trains agents to play the **larger** games and reports what happens to the
**known results**. Two phenomena are named as the things to report:

- **Survival of the weakest** — does it persist, strengthen or reverse as the field grows?
- **Deliberate missing** — is abstention ever chosen?

Building to the question the whole thing turns on: *the weakest player in a truel should miss — what
does the weakest player in a gruel do?*

**What this folder serves: all of it.** The three dials the research question names — **game variant,
reward profile, player count** — are steps 7, 8 and 9. Both phenomena are measured in every step that
trains anything, and the agent is checked against an exact answer wherever one exists. If only one
folder in this repository is read, it should be this one.

---

## The investigative approach

The folder is built as an investigation with four movements, and the order is the argument:

| | Movement | Why it comes here |
|---|---|---|
| **1** | **Build an agent** (steps 1–3) | You cannot investigate with an instrument you have not built. Three steps, each adding exactly one idea |
| **2** | **Build the instruments** (steps 4–5) | Measuring the wrong thing confidently is worse than measuring nothing. The instruments are calibrated against known answers *before* they judge anything |
| **3** | **Validate** (step 6) | Where an exact answer exists, match it or fail. Steps 6 and 9 are the two places a pass/fail is possible |
| **4** | **Vary one dial at a time** (steps 7–9) | The research question has three variables. Turn each one alone and measure what moves |

The discipline that holds it together: **each step changes one thing.** Steps 1 and 2 hold the game
fixed and change the algorithm. Step 3 changes the game by one player. Steps 7, 8 and 9 hold the agent
fixed and change the game. A step that changes two things cannot tell you which one mattered.

---

## What we are varying

### The game variants

| Variant | Values | What it changes |
|---|---|---|
| **Seating** | `random`, `ranked` | Which accuracy sits at which position. Under **ranked** the weakest is at position 0 and accuracies ascend, so the two best shots are always adjacent — they shoot each other and the weakest is protected. Under **random** a fresh permutation is drawn every episode, so that protection becomes a lottery |
| **Topology** | `circle`, `complete` | Who may shoot whom. At **N=3 they are the same game** — a triangle *is* the complete graph. On a circle the constraint does not even bind until N=6 |
| **Abstention** | always available | Action `2` is a deliberate miss. This matters for placing the project in the literature: this is the **random-order-with-abstention** variant, in which both shooting and non-shooting equilibria exist |

### The reward profiles

| Profile | Win | Eliminated | Kill | Survive a step |
|---|---|---|---|---|
| `terminal` | **+1.0** | 0 | 0 | 0 |
| `shaped` | **+1.0** | −1.0 | +0.1 | +0.001 |

`terminal` is the honest one — it is the objective the maths in folder 02 actually solves, and it is what
every validation uses. `shaped` adds hints to make learning easier, and the hints are not free: a player
who kills several opponents and then dies can bank more than the win itself pays. **Any finding that
appears only under `shaped` is a finding about the reward, not about the game.** That is why step 8
exists.

### The player counts

| N | What it is | Exact answer? |
|---|---|---|
| **2** | the duel. Left and right are the same opponent, so the only real choice is *shoot or abstain* | trivial |
| **3** | the truel. Circle == complete | **yes** — 0.29630 / 0.33333 / 0.37037, exactly 8/27, 9/27, 10/27 |
| **5** | the first genuinely graph-shaped case worth gating | **yes** — see below |
| **7** | the largest field this folder trains on, and the one where the exact answer stops being kind to the weakest player | **yes** — 0.15335 / 0.17343 / 0.14916 / 0.13761 / 0.12996 / 0.13217 / 0.12432 |
| 10, 15 | scale, where the method must change | **no** |

**At N=3 the exact answer is `shoot_strongest`.** The equilibrium and that heuristic are the same policy
here, which makes N=3 the right place to calibrate an instrument and the *wrong* place to test whether an
agent has learned something a heuristic could not.

**At N=5 the exact profile is not monotone**, weakest first:

| Accuracy | 0.4 (weakest) | 0.5 | 0.6 | 0.7 | 0.8 (strongest) |
|---|---|---|---|---|---|
| Exact survival | **0.24959** | 0.21073 | 0.17970 | **0.16194** | 0.19804 |

The weakest player does best — survival of the weakest — but the **second-strongest fares worst of
anyone**, and the strongest is only third. An agent producing a tidy monotone ordering has *not* matched
the equilibrium, however sensible that ordering looks. This is what makes N=5 a real test where N=3 is
only a sanity check.

**At N=7 the shape changes again**, and this is the count that makes the difference between matching an
answer and matching a heuristic visible. The weakest player is no longer even the best-off one — rank 1
is — and the worst seat is the strongest. `shoot_strongest` predicts the opposite at every point.

**Where an exact answer exists at all is a lookup, not a claim.** Folder `02-exact-solver` establishes the
recorded range by walking upward until the solver refuses, and records what stopped it. Beyond that range
validation is impossible and the work changes from *matching a number* to *describing a behaviour*, which
is what folder 04 does.

---

## Why PPO, and why MAPPO

Three steps, three algorithms, and each exists because the previous one runs out.

**Why not stay with REINFORCE?** Its gradient estimate uses the whole episode return with no baseline, so
its variance grows with episode length, and it throws each batch away after a single update. At N=2 that
is survivable and it is the clearest possible illustration of the policy-gradient idea. It stops being
survivable as soon as episodes lengthen.

**Why PPO?** Two things. A **learned value baseline** subtracts the part of the return that was going to
happen anyway, which cuts the variance REINFORCE suffers from. And the **clipped objective** lets you
take several gradient epochs on the same batch of episodes without the policy drifting so far that the
data no longer describes it — so each expensive batch of self-play is used several times instead of once.
With sparse rewards and short episodes, both matter.

**Why MAPPO?** Because PPO's critic is wrong here, and wrong in a way that gets worse with N. Two facts
about this game make it so:

- **The reward is per-agent and almost always zero.** Only one player wins. A critic estimating "how good
  is this state" has to answer *for whom*.
- **The actor is drawn at random with replacement**, so one agent may make eight decisions in an episode
  while another makes one. Each agent's experience is scattered irregularly through a shared history.

So a critic that sees only the acting agent's four numbers cannot distinguish a return that agent *earned*
from one it *inherited* because someone else happened to die. MAPPO's answer is a **centralised critic**
$V_\phi(s, i)$ — it takes the global state **and** which agent is being scored. Training is centralised,
execution stays decentralised: the critic sees everything while learning, and the policy still sees only
its own four numbers when it plays. That is the architecture this deliverable is about. Whether it is
needed is measured, not assumed: At N=3 and N=5 it is not what makes the agent work: step 10 trains the same agent with a critic that sees only the actor's four numbers, and that arm passes the same gates. At N=7, where both arms fail, the centralised critic's worst-rank ratio is 0.149 higher on average over five seeds.

Those are the short answers. **[`THEORY.md`](THEORY.md) is the long one** — the return, the policy gradient
and where it comes from, the baseline and why subtracting it leaves the gradient unbiased, the clipped
objective, the centralised critic, CTDE, and why measuring who an agent shoots is not an optional extra. It
states everything twice, once in ordinary language and once in the formal language, and each part points at
the function here that implements it. Read it if the question is *why* the mathematics is this shape rather
than what the folder measured.

---

## The progression, step by step

Nine folders, each standalone — see [Standing alone](#standing-alone) below. Read downwards: **every step
exists because the one before it left a specific question open, and each step's result is what makes the
next question askable.** The closing line of each entry is the question it hands on.

**1 — [`step1-reinforce/`](step1-reinforce/) · REINFORCE at N=2.** Begin with the smallest thing that is
still a game. In the duel, left and right are the same opponent, so the only real choice is *shoot or
abstain* and the policy has almost nothing to learn — which is the point, because any failure here is the
algorithm's and not the game's. The exact answer is derived in the notebook before training,
$W(p,q) = p/(p+q)$, and the agent reaches it to ratios of 0.990 and 1.005.
*What it leaves open:* the update is crude. Every decision in a losing episode is pushed down whether or
not it was sensible, so the learning curve wobbles.

**2 — [`step2-ppo/`](step2-ppo/) · PPO at N=2.** Hold the game *exactly* fixed and change only the
algorithm, so anything that moves is attributable. Two additions: a **value baseline**, so a decision is
judged against what was expected rather than against zero, and a **clipped objective**, so one batch of
self-play can be reused for several epochs without the policy running away from the data that described
it. Same survival answer as REINFORCE. Matched budgets show no difference in episodes needed: both reach 0.97 of exact on every seed from 320 episodes, because the duel is too easy to separate them. PPO is kept for the critic step 3 needs, not for speed.
*What it leaves open:* the critic sees one player's own view, and at N=2 that happens to be all there is.

**3 — [`step3-mappo/`](step3-mappo/) · MAPPO at N=3. The architecture.** Add one player and the critic's
job changes qualitatively, because *"is this state good?"* becomes *"good for whom?"*. The actor is drawn
at random with replacement, so a return can be inherited — a player can win having never fired — and a
critic seeing a single viewpoint cannot tell an earned return from a lucky one. Hence $V(s, i)$: the
critic is given the whole state **and** told which player it is scoring, then discarded after training.
Step 10 later tests whether this is needed.
The exact triple is reproduced with the **ordering correct**, all three within 3% of exact.
*What it leaves open:* "within 3%" was judged by reading numbers off a screen. There is no instrument.

**4 — [`step4-measurement/`](step4-measurement/) · the instruments.** Nothing that learned is judged here.
The step builds the measuring equipment — classification by stagnation **first**, survival by accuracy
rank, the ratio gate, standard errors and bootstrap intervals — and calibrates each one against policies
whose answers are known in advance. The refusal that matters most: `always_miss` must yield **no
targeting decisions at all** rather than a rate of `0.0`, because a rate of zero would assert it always
shot the weaker neighbour when in fact it never fired.
*What it leaves open:* every instrument so far measures **outcomes**, and outcomes cannot separate two
very different policies.

**5 — [`step5-targeting/`](step5-targeting/) · who does it shoot?** A flat survival profile is produced
both by a near-uniform equilibrium and by a policy discriminating on nothing. Survival numbers cannot
tell those apart; **who the agent aims at** can. The instrument is calibrated at N=3, where the
equilibrium *is* `shoot_strongest` so the answer is known before measuring — and then at N=5, because at
N=3 the survivor-count axis turns out to be **empty**: every countable decision has exactly three alive,
which is a property of the game rather than of the agent.
*What it leaves open:* everything to this point is calibration. Nothing has been asked to pass anything.

**6 — [`step6-n5-gate/`](step6-n5-gate/) · the N=5 gate. The first pass/fail.** Ranked seating against the
non-monotone profile above, five seeds. **PASS in 4 of 5**, at the threshold rather than clear of it: three
of the four passing seeds are within one evaluation standard error of 0.95. The single most informative line is not the
gate but the ordering: every seed puts the minimum at accuracy 0.7, 5 of 5. And targeting recovers both
the equilibrium's U-shape and its **rise as the circle contracts**, 5 of 5. Both are shared with a
fixed rule (Archetti's, restricted to the neighbours, which under ranked seating is "always shoot
left"); the agent is told apart from it by the size of the middle ranks and by firing at the stronger
neighbour almost every time with three left, which the fixed rule does 2 in 3. So the agent reproduced
the equilibrium's *policy*, not merely the survival numbers that policy produces.
*What it leaves open:* one configuration. One seating, one reward, one player count.

**7 — [`step7-seating/`](step7-seating/) · the first dial: seating.** Same agent, same N, same reward;
only which accuracy sits where. The two arms learn **different policies** — ranked matches the
equilibrium, random matches `shoot_strongest`. And the headline reading is a trap: the weakest survives
*better* under random seating, which taken from survival alone says "shuffling the seats helps the
weakest". It does not. The agent stopped finding the equilibrium and settled on a cruder rule that
happens to concentrate fire on whoever is strongest. Played in the ranked game, the random-trained models
are 0.0041 from `shoot_strongest`, so this is a different policy, not the same one measured in a different
game. **Step 5's instrument is what makes that visible**,
and this is the clearest case in the folder for why step 5 exists.
*What it leaves open:* the finding might be an artefact of the reward that produced it.

**8 — [`step8-reward-profile/`](step8-reward-profile/) · the second dial: reward.** `terminal` against
`shaped`, everything else held. **No measurable difference at N=5** — and the null is *predicted before
it is measured*, from what the kill term is worth as a fraction of the win at each player count. That
fraction grows with N, so the null is a statement about N=5 and not a claim that the reward never
matters. Only the terminal arm may be gated, because the exact answer is a set of survival
probabilities and a shaped agent is maximising something else — a guard in the code refuses the
comparison rather than trusting the reader to remember.
*What it leaves open:* every result so far is at one field size.

**9 — [`step9-player-count/`](step9-player-count/) · the third dial: N. The terminal step.** N=3, then 5,
then 7, each against its own recorded exact answer — and **the three answers are three different shapes**,
so an agent producing one shape three times has not tracked them. N=3 and N=5 pass; **N=7 fails in all
three seeds**, and the failure is legible: every middle rank sits below both extremes, which is true at
N=7 and false at the other two. The diagnosis is measured rather than inferred. Compared by rank with
the exact N=7 targeting, every middle rank fires at its stronger neighbour less often than the
equilibrium does, most of all near the top: the second-strongest at 0.3196 against about 0.54, the
strongest at 0.9999 against about 0.80. The agreement between the N=5 and N=7 runs at five and four
alive is weaker evidence than it looks, because a fixed always-shoot-left rule gives the same values
there (folder 04 step 3). This folder also carries
[the complete agent](step9-player-count/#the-agent-in-full), written out in full.

**10 — [`step10-critic-ablation/`](step10-critic-ablation/) · is the centralised critic needed?** The
reviewer's first question, answered by ablation: the same agent, seeds and budget at N=3, 5 and 7, with
the critic given either the global state and a one-hot (MAPPO) or only the actor's four numbers
(independent PPO). The centralised arm reproduces step 9 seed for seed. **Both arms get the same gate
verdict at every size**: pass at N=3 and N=5, fail at N=7. At N=3 and N=5 the paired differences are
small and in opposite directions (+0.010 and -0.023); at N=7 the centralised critic is 0.149 better on the
worst rank and closer in targeting, without passing a seed. One local run at N=7 fires right round the
circle instead of left; on the four seeds where both arms fire left the gap is still +0.146. Its value estimates are not better on
average: explained variance is slightly lower than the local critic's at every size.
*What it leaves open:* whether the N=7 advantage holds at larger N. Folder 04 step 8 tested it at 10 and 15
players without an answer key: the switch test finds no measurable difference between the critics.

**Steps 1–3 build. Steps 4–5 instrument. Step 6 validates. Steps 7–9 investigate. Step 10 tests the architecture.** The folder ends on a
step that runs and prints a result, and that result is a **failure with a readable cause** — which is a
better place to end than a clean pass, because it says what the method can and cannot yet do.

---

## A large gap against a heuristic is not a result

The equilibrium's weakest-over-strongest survival ratio **does not climb with N**: over N=3 to N=8 it
runs 1.25× / 1.25× / 1.26× / 1.66× / 1.23× / 0.96×, peaking at N=6 and then inverting — by N=8 the
weakest player is *worse* off than the strongest. The `shoot_strongest` heuristic's ratio climbs at
every single step over the same range: 1.25× → 2.31× → 3.69× → 5.40× → 7.45× → 9.84×. So a policy
showing a big, growing spread between weakest and strongest is displaying *that heuristic's
signature*, not evidence of good play. A success criterion rewarding a steep ratio would be
rewarding the agent for resembling a heuristic that stops being optimal above N=3.

This is why the investigation measures **who the agent shoots** rather than only how long it survives.

---

## Standing alone

Every step folder is self-contained, because a folder you cannot hand over on its own is a folder nobody
can read on its own:

- **Its own `gruel_env.py`** — the identical PettingZoo environment in every folder. Copy any single
  folder anywhere and it still runs.
- **Its own `requirements.txt`** — install from it into a fresh virtual environment and the folder works.
- **Its own `README.md`** — stating what the step does and **what number to expect** when you run it.
- **No imports between folders.** Not one. Where a step needs the previous step's code, it holds **its own
  copy**, so you read a step top to bottom without opening another folder. Duplication is the price of
  that, and it is worth paying here.

**The environment is the one folder 01 built** — the PettingZoo AEC class `GruelEnv`. Every training loop
in this folder is the ordinary PettingZoo loop:

```python
for acting_agent in game.agent_iter():
    observation, reward, termination, truncation, info = game.last()
    if termination or truncation:
        game.step(None)
        continue
    game.step(chosen_action)
```

There is no second implementation of the game anywhere in this repository.

## Running a step

```bash
cd step1-reinforce
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/jupyter lab step1_reinforce.ipynb
```

Each notebook ends by running the same thing twice from the same seed and printing whether the two
results are identical. A step that cannot reproduce itself cannot support a claim.
