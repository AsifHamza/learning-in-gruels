# Step 3 — Who the agent shoots

Steps 1 and 2 look at outcomes: who survives, and whether a player could do better by switching. Outcomes cannot say what a policy does, because two different policies can produce the same survival numbers. This step measures behaviour directly.

Every time a player fires, it fires at one of its two neighbours. The question is: **how often does it pick the stronger one?** Read by the player's own strength and by how many players are left, that one number separates policies that survival cannot.

## What counts as a decision

A decision counts only if the question has an answer. Three cases are excluded:

1. **The player abstained.** It chose no target.
2. **Two players are left.** Both neighbours are the same opponent.
3. **The two neighbours are equally strong.** There is no stronger one.

When nothing is left to count, the rate is reported as missing, not as zero. Zero would claim the player always shot the weaker neighbour.

## Four patterns to read the agent against

| Pattern | Ends of the ranking | Middle | Depends on players left? |
|---|---|---|---|
| matches the exact equilibrium | about 1.0 | about 0.3 to 0.4 at N=5 | strongly |
| copies `next_weaker` | 1.0 | low, lower as N grows | yes, as 2 in k |
| copies `shoot_strongest` | high | high | no |
| shoots at random | about 0.5 | about 0.5 | no |

`next_weaker` is Archetti's rule restricted to the two neighbours, the same projection folder 02 step 5 measures: shoot the strongest neighbour who is still weaker than you, and if both are stronger, shoot the stronger one. **It is here because it is a fixed rule that already has the U-shape.** Under `ranked` seating the circle stays sorted as players die, so the next weaker player is always the one on your left, and the weakest player's left-hand neighbour is the strongest. The rule is therefore the same as "always shoot left", and with $k$ players left it hits the stronger neighbour at exactly two seats, the weakest and the strongest. The notebook checks both facts: `next_weaker` shoots left in 1.0000 of its shots at every size, and its rate by players left is within 0.0065 of $2/k$. So under this seating any policy that mostly fires one way round the circle looks U-shaped by rank, whatever it knows about strength.

The middle column separates the equilibrium from `shoot_strongest` and `coin_flip`; only the size of the middle values separates it from `next_weaker`. All three references are measured in the same game, so the references come from this folder rather than from memory. A reference is named as a match only if it is within 0.10 on average; otherwise the answer is "neither".

## What number to expect

`ranked` seating, `terminal` reward, trained exactly as in step 1 (2000 updates × 96 episodes, seeds 0 to 2), 10,000 measured episodes per seed. About 40 minutes on a laptop CPU, almost all of it training; wall-clock time varies.

### Checked first at five players

Folder 02 knows the exact equilibrium's targeting at five players, so the measurement is checked there before it is read anywhere else.

| | rank 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| agent, pooled | 0.9998 | 0.3777 | 0.3163 | 0.3781 | 0.9998 |
| exact | 1.0000 | 0.3979 | 0.3107 | 0.3714 | 1.0000 |

The agent is 0.0066 from the exact answer on average and matches it, against 0.1160 from `next_weaker`, whose middle ranks are 0.1986, 0.0935 and 0.2005. The exact row weights each state by how often a shooter makes a decision there, which is how the agent is measured; folder 02 step 5 prints both weightings. By players left it gives 0.3991, 0.5159 and 0.9777 at five, four and three, against the exact 0.40, 0.50 and 1.00, a worst gap of 0.0223, where `next_weaker` gives 2 in 3 with three left. **The measurement recovers the exact equilibrium where it is known, and at five players it tells the equilibrium apart from the fixed rule.**

### At ten and fifteen players

```
N=10   0.9998 0.3667 0.2084 0.1554 0.1361 0.1376 0.1599 0.2066 0.3580 0.9991
N=15   0.9999 0.3719 0.1999 0.1289 0.1047 0.0887 0.0822 0.0871 0.0952 0.1034
       0.1208 0.1589 0.2121 0.3573 0.9999
```

**At ten and fifteen players the agent is `next_weaker`.** Its targeting by rank is 0.0424 from `next_weaker` at ten players and 0.0225 at fifteen, inside the 0.10 threshold, so the verdict at both sizes is "matches `next_weaker`". The deep U, falling to 0.0822 in the middle at fifteen players, is what always shooting left produces under ranked seating; it is not evidence that the agent found something deeper than the five-player equilibrium. All three seeds show the same shape: at fifteen players the middle minimum per seed is 0.0460, 0.0785 and 0.1141. The other references are far away: `coin_flip` 0.3255 and 0.3603, `shoot_strongest` 0.6272 and 0.7259.

**Seeds 0 to 2 all fire left, but not every seed does.** Step 8 trains five seeds, and four of its twenty runs at 10 and 15 players fire right round the circle instead: each player shoots the next stronger one and the strongest shoots the weakest, the mirror image of `next_weaker`. So "the agent is `next_weaker`" describes the common outcome at these sizes, and three of three seeds here, not every training run.

So the five-player agent and the larger agents are different in kind. At five players the agent's middle ranks sit near the equilibrium's and well above `next_weaker`'s. At ten and fifteen they sit on `next_weaker`. Whether the equilibrium at these sizes looks like `next_weaker` is unknown, because no exact answer is on record above eight players.

### It depends on how many players are left

| players left | 15 | 12 | 10 | 8 | 6 | 5 | 4 | 3 |
|---|---|---|---|---|---|---|---|---|
| started with 15 | 0.1294 | 0.1682 | 0.1996 | 0.2496 | 0.3333 | 0.4168 | 0.5922 | 0.7977 |
| started with 10 | – | – | 0.1978 | 0.2497 | 0.3328 | 0.4026 | 0.5624 | 0.9286 |

Early in a large game the agent seldom fires at the stronger neighbour; as players drop out it does so more and more, reaching 0.80 to 0.93 with three left. Most of that rise is `next_weaker`'s $2/k$: at fifteen players the agent is within 0.0182 of `next_weaker` at every count from fifteen left down to five, and at ten players within 0.0091. It departs only at the end, by 0.0583 and 0.2660 at ten players with four and three left, and by 0.0899 and 0.1349 at fifteen. With three left, `next_weaker` fires at the stronger neighbour 2 times in 3 and the agent 0.80 to 0.93 of the time, moving toward the equilibrium's three-player answer of always. Step 4 compares these curves across sizes.

## The upper-middle players, read against step 2

Step 2 found that at fifteen players the strong players short of the strongest would win more often by switching to `shoot_strongest`. Their targeting here shows why that switch is such a large change:

| rank | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|
| agent | 0.0952 | 0.1034 | 0.1208 | 0.1589 | 0.2121 | 0.3573 |

Averaged over those ranks the agent fires at the stronger neighbour 0.1746 of the time, where `shoot_strongest` does so every time. So the two steps agree: these players mostly shoot their weaker neighbour, and step 2 shows they would do better shooting the stronger one.

**These runs use `ranked` seating**, and deliverable 03's step 7 showed at five players that seating changes which policy is learned: random seating led the agent to `shoot_strongest` and ranked seating to the equilibrium's shape. The equivalence between `next_weaker` and "always shoot left" also holds only under ranked seating, so the deep U here is a result about ranked seating twice over.

## Reproducibility

Two random streams are reset, not one: the environment's, and torch's global stream, because a trained policy samples its actions. The notebook measures the same thing twice and prints whether the results are identical. They are.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook step3_targeting.ipynb
```

This folder stands alone: it carries its own copy of the environment and its own pinned requirements, and imports nothing from any other step.

**Next:** [`../step4-continuity/`](../step4-continuity/)
