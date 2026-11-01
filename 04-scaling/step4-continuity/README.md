# Step 4 — Is it still the same strategy?

This is the last step of deliverable 04, and the one to run if only one is run.

Above eight players nothing can compute the right answer, so the agent at 10 and 15 players cannot be checked against one. It can be checked against **itself**, across the range where it *was* checked. The comparison is made at the **same number of players left**: a game that starts with fifteen players eventually has seven left, then five, then three. If the agent at that moment behaves as the seven-player agent does with the same number left, it is following a rule about its position in the game rather than about how large the game started.

Neither side of the comparison needs an exact solution, because both are measurements of this project's own agent.

## The measurement

The share of decisions in which the shooter fired at its stronger neighbour, by how many players were left. A decision counts only if the player fired, more than two players were left, and its two neighbours differed in strength. Two measurements **agree** if they differ by at most 0.10, the same threshold deliverable 03 uses.

## What number to expect

All five player counts, `ranked` seating, `terminal` reward, trained exactly as in step 1 (2000 updates × 96 episodes, seeds 0 to 2), 10,000 measured episodes per seed. About 50 minutes on a laptop CPU; wall-clock time varies.

Each cell pools three seeds; the ± is half the range across seeds.

| players left | started with 3 | 5 | 7 | 10 | 15 |
|---|---|---|---|---|---|
| 15 | | | | | 0.1294 ±0.01 |
| 12 | | | | | 0.1682 ±0.00 |
| 10 | | | | 0.1978 ±0.00 | 0.1996 ±0.00 |
| 9 | | | | 0.2226 ±0.00 | 0.2268 ±0.00 |
| 8 | | | | 0.2497 ±0.00 | 0.2496 ±0.01 |
| 7 | | | 0.2822 ±0.00 | 0.2861 ±0.01 | 0.2830 ±0.00 |
| 6 | | | 0.3338 ±0.00 | 0.3328 ±0.00 | 0.3333 ±0.00 |
| 5 | | 0.3991 ±0.00 | 0.3976 ±0.00 | 0.4026 ±0.00 | 0.4168 ±0.03 |
| 4 | | 0.5159 ±0.01 | 0.5075 ±0.01 | 0.5624 ±0.05 | 0.5922 ±0.10 |
| 3 | 0.9983 ±0.00 | 0.9777 ±0.01 | 0.8426 ±0.15 | 0.9286 ±0.04 | 0.7977 ±0.11 |

## The result: one curve, at every size, and most of it is a fixed rule

**Read across any row and the numbers barely move.** With seven players left, the agents that started with 7, 10 and 15 players fire at the stronger neighbour 0.2822, 0.2861 and 0.2830 of the time. With six left, 0.3338, 0.3328 and 0.3333. With eight, the 10- and 15-player agents give 0.2497 and 0.2496. Against the seven-player agent, both the 10- and the 15-player agent agree at all five numbers of players left they share.

**That agreement is mostly not about the agent.** Step 3 introduced `next_weaker`, Archetti's rule restricted to the two neighbours, which under `ranked` seating is the same as always shooting left. With $k$ players left it hits the stronger neighbour at two of the $k$ seats, so its value in this table is $2/k$ at every starting size: measured here, 0.2847, 0.2845 and 0.2894 with seven left, against $2/7 = 0.2857$. A fixed rule that knows nothing about the starting size draws the same rising curve. The agent sits on it: from the start of the game down to five left, every cell of the table is within 0.0182 of `next_weaker` at the same starting size.

| players left | `next_weaker` ($2/k$) | agent minus `next_weaker`, started with 3 / 5 / 7 / 10 / 15 |
|---|---|---|
| 7 | 0.2857 | – / – / -0.0025 / +0.0016 / -0.0064 |
| 5 | 0.4000 | – / +0.0005 / -0.0076 / +0.0091 / +0.0182 |
| 4 | 0.5000 | – / +0.0164 / +0.0044 / +0.0583 / +0.0899 |
| 3 | 0.6667 | +0.3292 / +0.3096 / +0.1816 / +0.2660 / +0.1349 |

So a comparison between sizes says something about the agent only where at least one side is more than 0.10 from `next_weaker`. The notebook marks those counts. Against the seven-player agent there is one for each larger size: **three left**. There the 10- and 15-player agents fire at the stronger neighbour 0.9286 and 0.7977 of the time, against 0.8426 at seven, and all three are well above `next_weaker`'s 2 in 3. That one count is the evidence that the agent's own departure from the fixed rule carries across sizes.

**Three left is also where the measurement is noisiest.** Between neighbouring sizes two pairs differ by more than the threshold there, five against seven (0.1351) and ten against fifteen (0.1309), and the spread *between seeds of the same size* is as large (±0.15 for seven players with three left, ±0.11 for fifteen). So the carried departure is real in direction, since every size is above 2 in 3, but its size is not pinned down.

## Where the truth is known: is that rule right?

Continuity is only worth something if the range it continues from was right. So each checked size is compared with its own exact answer first.

| size | players left | agent | exact |
|---|---|---|---|
| 3 | 3 | 0.9983 | 1.0000 |
| 5 | 5 / 4 / 3 | 0.3991 / 0.5159 / 0.9777 | 0.4000 / 0.5000 / 1.0000 |
| 7 | 7 / 6 / 5 / 4 / 3 | 0.2822 / 0.3338 / 0.3976 / 0.5075 / 0.8426 | 0.1429 / 0.4517 / 0.4000 / 0.5000 / 1.0000 |

**The rule is right at three and five players and not at seven.** At seven the exact equilibrium fires at the stronger neighbour 0.1429 of the time with all seven left and then *jumps* to 0.4517 with six; the agent instead climbs smoothly, 0.2822 then 0.3338. Its worst gap at seven players is 0.1574.

Read with `next_weaker` alongside, the table says this. At five players the agent's 0.40 and 0.50 coincide with both the equilibrium and the fixed rule, and with three left it fires at the stronger neighbour 0.9777 of the time, like the equilibrium and unlike the fixed rule. At seven the equilibrium departs from $2/k$ early (0.1429 with seven left, 0.4517 with six); the agent does not, and stays on the fixed rule until the last three. At ten and fifteen it does the same.

## What this does and does not establish

**It establishes** that at every size from five to fifteen the agent fires one way round the circle, as `next_weaker` does, until three players are left, and then fires at the stronger neighbour far more often than that rule does. The five-player check is the one place where this behaviour is confirmed as equilibrium play. At seven players the same behaviour misses the equilibrium, which departs from the fixed rule earlier in the game.

**It does not establish** that the agent has learned a rule about strength that carries over, as opposed to a direction to fire in. Most of the agreement between sizes is the agreement of a fixed rule with itself. Nor does it establish that the policy is right at 10 or 15. Continuity is not correctness: an agent can apply the same wrong rule at every size and look perfectly continuous, and step 2 has already shown that at fifteen players some players would do better by switching. Nor is "five players left" the same situation in every game: five left from fifteen are more closely matched in strength than five left from seven, because the starting field was packed more tightly. The agent behaving identically in those two situations is what it does, not proof that it should.

## Reproducibility

Two random streams are reset, not one: the environment's, and torch's global stream, because a trained policy samples its actions. The notebook measures the same thing twice and prints whether the results are identical. They are.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook step4_continuity.ipynb
```

This folder stands alone: it carries its own copy of the environment and its own pinned requirements, and imports nothing from any other step.

Steps 5 to 7 then turn the reward and seating dials at 10 and 15 players, and put an error bar on how often a run holds fire.
