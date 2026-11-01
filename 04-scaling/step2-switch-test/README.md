# Step 2 — Can one player do better by switching?

Folder 02 defines an equilibrium as a set of strategies where **no single player can do better by changing what it does on its own**. That is a test, and it does not need the exact answer to run. This step runs it on the trained agent at 5, 10 and 15 players.

## The test

1. Everyone plays the trained agent. Record each seat's expected return.
2. One seat at a time switches to one of four simple rules — `shoot_strongest`, `shoot_weakest`, `coin_flip`, `always_miss` — while everyone else keeps playing the agent. Record that seat's return again.
3. If switching clearly raises the seat's return, the agent's policy is **not** an equilibrium: that player has a reason to abandon it.

Under the `terminal` reward a seat's return is its chance of winning, so "does better" means exactly what it means in folder 02.

**What a result means.** A profitable switch is a proof: the policy is not an equilibrium. No profitable switch is evidence, not proof, because an equilibrium must resist every possible strategy and only four are tried.

**What counts as clearly.** A gain must exceed three standard errors. The notebook runs 360 switch tests, so a two-standard-error threshold would flag about 8 by chance alone; at three, the expected number of false alarms across the whole notebook is 0.49.

## Checking the test before trusting it

At five players folder 02 knows the answer, so the test is checked against it twice before it is used where nothing can check it.

**It finds a profitable switch where one exists.** When everyone plays `shoot_strongest`, folder 02 shows that is not an equilibrium. The test agrees: the seat with accuracy 0.6 raises its chance of winning from 0.0890 to 0.1402 by switching to `shoot_weakest`, a gain of +0.0512. Two profitable switches are found in total.

**It finds none where none should exist.** At five players the trained agent matches the exact equilibrium (deliverable 03, step 6). Across all three seeds, 0 of 60 switches are profitable.

## What number to expect

`ranked` seating, `terminal` reward, trained exactly as in step 1 (2000 updates × 96 episodes, seeds 0 to 2). Each seat's baseline is measured over 10,000 episodes and each switch over 5,000. The whole step runs in about 65 minutes on a laptop CPU, most of it training; wall-clock time varies from run to run.

| N | profitable switches, per seed | of |
|---|---|---|
| 5 | 0 / 0 / 0 | 20 each |
| 10 | 0 / 0 / 1 | 40 each |
| 15 | 6 / 5 / 1 | 60 each |

So at five players no seed's policy can be improved by any switch tried. At ten, two of three seeds survive every switch. **At fifteen, every seed's policy has at least one profitable switch, so none of them is an equilibrium.**

## Who gains, and by switching to what

All thirteen profitable switches, from the notebook's output:

| N | seed | seat | accuracy | switches to | return before → after | gain | standard errors |
|---|---|---|---|---|---|---|---|
| 10 | 2 | 8 | 0.7556 | `shoot_strongest` | 0.0741 → 0.1004 | +0.0263 | 5.3 |
| 15 | 0 | 8 | 0.6286 | `shoot_strongest` | 0.0509 → 0.0654 | +0.0145 | 3.5 |
| 15 | 0 | 9 | 0.6571 | `shoot_strongest` | 0.0543 → 0.0680 | +0.0137 | 3.2 |
| 15 | 0 | 11 | 0.7143 | `shoot_strongest` | 0.0544 → 0.0782 | +0.0238 | 5.4 |
| 15 | 0 | 11 | 0.7143 | `coin_flip` | 0.0544 → 0.0678 | +0.0134 | 3.2 |
| 15 | 0 | 12 | 0.7429 | `shoot_strongest` | 0.0589 → 0.0902 | +0.0313 | 6.7 |
| 15 | 0 | 13 | 0.7714 | `shoot_strongest` | 0.0804 → 0.1150 | +0.0346 | 6.6 |
| 15 | 1 | 10 | 0.6857 | `shoot_strongest` | 0.0433 → 0.0558 | +0.0125 | 3.3 |
| 15 | 1 | 11 | 0.7143 | `shoot_strongest` | 0.0398 → 0.0532 | +0.0134 | 3.6 |
| 15 | 1 | 12 | 0.7429 | `shoot_strongest` | 0.0400 → 0.0596 | +0.0196 | 5.1 |
| 15 | 1 | 13 | 0.7714 | `shoot_strongest` | 0.0523 → 0.0770 | +0.0247 | 5.6 |
| 15 | 1 | 13 | 0.7714 | `coin_flip` | 0.0523 → 0.0648 | +0.0125 | 3.0 |
| 15 | 2 | 13 | 0.7714 | `shoot_strongest` | 0.0314 → 0.0500 | +0.0186 | 5.3 |

**The pattern is consistent across seeds.** Every profitable switch is made by a strong player that is not the strongest: seats 8 to 13 of 15, and seat 8 of 10. Eleven of the thirteen switch to `shoot_strongest`, and the two others to `coin_flip`. No profitable switch is made by any of the weaker half of the field, and none by the single strongest player.

So the agent leaves the upper-middle of the field playing worse than a simple rule would: those players would do better by firing at their stronger neighbour every time. Because seating is ranked, that neighbour is the next player up, ending at the strongest.

The same pattern shows up, more weakly, in the average best gain per seat: at 15 players it runs close to zero for the weaker seats and rises to +0.0147, +0.0202 and +0.0260 for seats 11, 12 and 13.

## How this fits the rest of the project

At five players the agent passes this test, as it should, since it matches the exact equilibrium. Deliverable 03's step 9 found that at seven players it no longer matches the exact answer. This step finds that at ten and fifteen it is measurably not an equilibrium, at sizes where nothing else could have said so.

It does not say *how far* from an equilibrium the agent is, or what an equilibrium at fifteen players would look like. It says that one specific improvement is available, to whom, and by how much.

## Reproducibility

Two random streams are reset, not one: the environment's, and torch's global stream, because a trained policy samples its actions. The notebook ends by measuring the same switch twice and printing whether the results are identical. They are.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook step2_switch_test.ipynb
```

This folder stands alone: it carries its own copy of the environment and its own pinned requirements, and imports nothing from any other step.

**Next:** [`../step3-targeting/`](../step3-targeting/)
