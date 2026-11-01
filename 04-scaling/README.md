# Deliverable 04 — The agent beyond the answer key

**What this deliverable is:** take folder 03's agent to **10 and 15 players**, past the point where
anything can compute the right answer, and judge it without one.

**[`GAME.md`](GAME.md) sets the game out end to end** — how players are counted, how accuracies are
assigned, how the circle contracts, what the two reward profiles pay, and the three ways an episode can
end. Read it first if you are reading a number here and want to know what was actually played.

---

## What this folder answers to

**The brief.** Gruels generalise truels to a graph; exact strategies are computable only in small
cases; and the project trains agents to play the **larger** games and reports what happens to the
**known results**. Two phenomena are named as the things to report:

- **Survival of the weakest** — does it persist, strengthen or reverse as the field grows?
- **Deliberate missing** — is abstention ever chosen?

Building to the question the whole thing turns on: *the weakest player in a truel should miss — what
does the weakest player in a gruel do?*

**What this folder serves: the word "larger".** The brief asks for agents playing the *larger* games,
and this is the largest the project attempts. Both phenomena are still the things to report — the
difference is that here they are reported without an answer key to check them against, which is a
weaker kind of statement and is labelled as one throughout.

**A flat survival profile is not evidence about the game.** An agent whose targeting is
undifferentiated produces a flat profile because it discriminated on nothing, so its flatness is a fact
about the policy. Only a policy that learned something can be evidence about the phenomenon, which is
why step 3 measures who the agent shoots rather than reading survival alone.

---

## The single most important thing about this folder

**It contains almost no new code.** It is folder 03's trainer with a bigger `N`.

The action space doesn't even change: `Discrete(3)` — shoot left, shoot right, miss — at *every* player
count, because on a circle you always have exactly two neighbours. `N` is a number at the top of a
config file.

What changes is that **the answer key has run out.**

| N | States to solve | Strategy combinations | Rough total checks | Exact answer |
|---|---|---|---|---|
| 3 | 4 | 27 | 108 | yes, instant |
| 5 | 26 | 243 | 6,318 | yes, ~0.004 s |
| 7 | 120 | 2,187 | 262,440 | yes, ~0.1 s |
| 8 | 247 | 6,561 | 1,620,567 | yes |
| 9 | 502 | 19,683 | 9,880,866 | **no — no pure equilibrium exists** |
| 10 | 1,013 | 59,049 | 59,816,637 | **no — refused at a state with 8 players alive** |
| 15 | 32,752 | 14,348,907 | 469,955,402,064 | **no — refused at a state with 8 players alive** |

Timings measured in [`../02-exact-solver/step3-circle-n5/`](../02-exact-solver/step3-circle-n5/);
counts are arithmetic. Roughly a factor of 5 per player added — but that growth is not what
ends the exact approach. Carried forward from N=7 it puts N=10 near 13 seconds, which is
affordable. **The wall is at N=9, where a state has no pure Nash equilibrium**, so there is nothing
for backward induction over pure strategies to select. The same happens at 10 and 15 players: the
solver refuses at a state with eight players alive, after about three minutes at 15
(`../02-exact-solver/step4-how-far`, section 4.7). The refused N=9 state does have a mixed equilibrium,
so a solver over mixed strategies could go further; this project does not build one. So no game this
folder trains at has an exact answer on record.

**This table is the project's central argument.** Folders 01–03 could all be checked against
arithmetic. This one cannot be, and that is precisely the situation the project exists to address: if
you cannot compute the answer, can you learn a good policy anyway? The answer key stops because the
solver's method runs out, not because it runs out of time.

---

## Which runs come here

Folder 03's agent is the one that scales, seeing only its two neighbours. At 15 players the
graph is at its most restrictive — an agent sees four numbers and has fourteen opponents, only two of
them reachable — so this is where the question of whether the approach scales is answered.

**This folder's own runs.** Trained here, from scratch, with the same code folder 03 uses:

| N | Seating | Reward | Seeds | Why |
|---|---|---|---|---|
| 3, 5, 7 | `ranked` | `terminal` | 3 | re-measured at this budget so the trend is one experiment, and they still have an answer key |
| **10, 15** | `ranked` | `terminal` | 3 | **the new work** — past the point where any answer key exists |

`ranked` seating and the `terminal` reward, because three of those five counts have an exact answer and
those are the two conditions under which an exact answer may be read. `terminal` also makes a survival
probability *be* the expected return, which is what lets the two be compared at all.

---

## There is no gate at this scale — and that is the point

Folders 01, 02 and 03 could all be checked against an answer key. **This one cannot**, and the honest
consequence is that N=15 carries **no pass/fail gate at all**. The project's gates live at N=3 and N=5,
where the solver can compute the truth. Above N=8 nothing can.

So what replaces a gate is a **description** of what the agent does, plus the one test that needs no
answer key at all.

**That test is folder 02's definition of an equilibrium, applied directly.** An equilibrium is a set of
strategies where no single player can do better by changing what it does on its own. So step 2 lets
everyone play the trained agent, then lets one seat at a time switch to one of four simple rules —
`shoot_strongest`, `shoot_weakest`, `coin_flip`, `always_miss` — and asks whether that seat does
better. A switch that clearly helps proves the agent's policy is not an equilibrium. No switch helping
is evidence, not proof, because only four alternatives are tried.

**Why it is per seat** and not overall: overall survival is pinned at $1/N$ by construction — exactly
one player survives, so the mean across players cannot be improved by anyone. A switch can only move
one seat's share.

**Why beating a baseline is not a criterion.** A large gap against `shoot_strongest` is that
heuristic's signature rather than evidence of good play. Where the truth is computable, the
equilibrium's weakest-over-strongest ratio does not climb: it runs 1.25x, 1.25x, 1.26x, 1.66x, 1.23x
and 0.96x from N=3 to N=8, peaking at N=6 and ending below 1 — while `shoot_strongest`'s climbs at
every step, 1.25x to 9.84x over the same range. So a steep ratio at N=15 says the agent
resembles the heuristic, not that it plays well. A criterion that rewards it would reward the wrong
thing.

**What is measured instead** is *who the agent shoots*: the probability it fires at the stronger of its
two neighbours, broken down by its own accuracy rank and by how many players remain. That separates
cases a survival profile cannot. Folder 02 covers why.

---

## Why the reward here is `terminal`, and what that gives up

This folder trains under **`terminal`**: +1 for winning, nothing for anything else. Two reasons, and
they are the same reason twice. It is the objective the exact solver maximises, so three of the five
player counts can still be read against an answer. And under it a player's survival probability **is**
its expected return, so the quantity measured here and the quantity folder 02 computes are the same
quantity rather than two things that merely correlate.

**What it gives up is real.** The win signal is sparse at this scale — one winner in fifteen means a
1-in-15 chance of ever seeing the +1 — and `shaped` exists precisely to thin that out with hints.

**That is a question rather than a problem, and it is the sharpest one left open here.** Folder 03's
step 8 turned the reward dial at 5 players and measured **no difference that survived its error bars**.
It also said why, before measuring: the hints are worth what the kill term can reach as a fraction of
the win, which is 0.30 at 5 players and **1.30 at 15**. So the configuration where shaping should bite
hardest is exactly the one this folder trains at. Step 5 tests the prediction there, and finds that
shaping does not change survival at either 10 or 15 players.

A kill term of 1.30 exceeding the win reward of 1.0 does not mean losing pays better than winning: the
shaped profile also charges the victim -1.0 for being eliminated, so a prolific killer who dies nets 0.3
against a winner's 1.0. What shaping changes is the **gradient**, not the ordering — a kill is a certain,
immediate +0.1 while the win is distant and uncertain.

One value worth knowing about the shaped profile, because it explains a number that otherwise looks
arbitrary: $r_\text{survive}$ is **0.001 and not 0.01**. At 0.01, under a discount factor of 0.99,
the survival stream sums to the same total as the win reward — so stagnating would pay as well as
winning, manufacturing the exact failure the shaping was meant to prevent.

---

## What this folder found

**The weakest player's advantage survives past the answer key, but does not climb the way a simple rule's
does.** At fifteen players the weakest wins 1.517 times an even share, against 6.769 for a field playing
`shoot_strongest` (step 1).

**At ten and fifteen players the agent fires one way round the circle.** Its targeting by rank is 0.0424
and 0.0225 from `next_weaker`, Archetti's rule restricted to the neighbours, which under ranked seating is
the same as always shooting left (step 3). That rule already has the U-shape, so the deep U at fifteen
players (0.0822 in the middle) comes from the seating, not from a deeper reading of strength. At five
players the agent is 0.0066 from the equilibrium and 0.1160 from `next_weaker`, so there it does more than
the fixed rule.

**It is not an equilibrium at fifteen players.** In every seed, some strong players short of the strongest
would win more often by switching to `shoot_strongest` — up to +0.0346 for one seat (step 2). At five
players no switch helps.

**The curve by players left is the same at every size, and most of it is the fixed rule.** With seven
players left, the agents that started with 7, 10 and 15 players fire at the stronger neighbour 0.2822,
0.2861 and 0.2830 of the time, where `next_weaker` gives 2/7 at any size (step 4). Down to five left
every size is within 0.0182 of `next_weaker`. The agent departs from it only at the end: with three left
it fires at the stronger neighbour 0.7977 to 0.9777 of the time against the rule's 2 in 3, at every size.
That departure is what carries across sizes; it matches the equilibrium at five players, and at seven the
equilibrium departs from the fixed rule earlier than the agent does.

**One run in ten, at fifteen players, holds fire early.** It misses deliberately in 0.2976 of its
decisions, almost all while most of the field is still alive (step 1). Ten seeds put the rate at 0.10,
with an exact 95% interval of 0.0025 to 0.4450 (step 7), which still contains step 1's 1 in 3.

**The reward profile does not change survival, even where the hints are largest.** At 10 and 15 players
the weakest player's advantage under `shaped` is within twice the seed-to-seed error of `terminal`
(step 5). The policies match at 15 players and differ only slightly at 10.

**Seating changes the policy at every size.** Under random seating the agent's targeting flattens to
about three shots in four at the stronger neighbour, at every rank, where ranked seating gives the deep
U. Unlike at five players, the flat rule no longer matches `shoot_strongest` (step 6).

**The centralised critic shows no measurable benefit at 10 and 15 players.** Trained with a critic given
only the actor's four numbers, on the same five seeds, the agent is no more and no less exploitable on
the switch test, and the local critic predicts the return slightly better (step 8). So deliverable 03's
finding that the centralised critic helps at N=7 does not carry to these sizes.

**The agent learns one of two mirror-image policies, and the endgame decides the weakest player's fate.**
Over twenty seeds at each of 7, 10 and 15 players, 8 of 60 runs fire *right* round the circle (`next_stronger`:
each player shoots the next stronger one, the strongest the weakest), a rate of 0.13 with an exact 95%
interval of 0.0594 to 0.2459 and no measurable dependence on the player count (step 9). Under that policy the
strongest seat pays: at 15 players it survives 0.4819 times an even share against 1.5288 under left-shooting.
Among left-shooting runs, those that keep to `next_weaker` to the end give the weakest about what the rule
gives it, and those that turn on the stronger neighbour near the end give it much more, +0.567 at 15
players (step 9). Steps 1 to 7 use seeds 0 to 2, which all fire left.

---

## The steps

| Folder | What it teaches | What stands in for an answer key |
|---|---|---|
| [`step1-train-any-n/`](step1-train-any-n/) | The same agent at 10 and 15 players, and the trend across five counts | three of the five counts can still be checked; two cannot |
| [`step2-switch-test/`](step2-switch-test/) | Whether any one player could do better by switching to a simple rule | folder 02's definition of an equilibrium, tested directly |
| [`step3-targeting/`](step3-targeting/) | Who the agent shoots, read against the three signatures | identification — a second statistic, not a grade |
| [`step4-continuity/`](step4-continuity/) | Whether the policy that worked at 7 players is the one still being played | continuity with the range where the truth is known |
| [`step5-reward-profile/`](step5-reward-profile/) | Whether the shaped reward changes survival or policy at 10 and 15 players | the two reward arms compared with each other |
| [`step6-seating/`](step6-seating/) | Whether random seating changes the policy at 10 and 15 players | the two seating arms compared with each other, and with two references |
| [`step7-holding-fire/`](step7-holding-fire/) | How often a run at 15 players holds fire, over ten seeds | a rate with an exact binomial interval |
| [`step8-critic-ablation/`](step8-critic-ablation/) | Whether the centralised critic matters at 10 and 15 players | the two critics compared on the switch test, with five seeds each |
| [`step9-direction/`](step9-direction/) | How often each of the two mirror-image policies is learned, and what the difference does | twenty seeds per count, with exact binomial intervals; the exact answer at N=7 |

All nine steps are built and executed. Step 4 is the one to run if only one is run. Steps 5 and 6 turn
the two dials deliverable 03 turned at five players, and step 7 puts an error bar on step 1's
holding-fire run. Step 8 is a script plus an analysis notebook, because its training runs for more than
an hour in sequence; it asks deliverable 03 step 10's question where no answer key exists.

## Two honest limitations

**Nothing here can be verified against arithmetic.** Every number is a comparison between policies. If
the trainer had a subtle bug that survived folder 03's gates, this folder could not detect it. That is
why those gates exist and why both had to pass first — and both now have.

**"Survives four switches" is a weaker claim than "found the equilibrium."** Folder 03 could say agents
reached the profile no player wants to deviate from, because folder 02 computed that profile exactly and
checked every deviation. Here only four alternatives can be tried, so the strongest positive claim is
that no player gains by switching to any of them. Stating the difference plainly is more defensible
than implying the earlier standard carries over.

---

## What this folder will deliberately not do

- **No sensitivity sweep.** 11 configurations × 3 seeds varying one factor at a time — a robustness
  measure, not a gate, and not part of this folder.
- **No non-circle topologies.** The circle is the game throughout.
- **No exact solver comparison.** By definition — that is what this folder is about.

This is the last deliverable.
