# Step 2 — The whole game, with no framework at all

## What this step is

The **rules**, in plain Python. One class, numpy, and nothing else. No Gymnasium, no
PettingZoo.

This is deliberately the wrong way round. Step 3 wraps this same game in a framework, and the
only honest way to understand what a framework is *for* is to already know what it is wrapping.

The rules in full:

1. One living player is picked **at random** to act — not in turn order, and the same player can
   be picked twice in a row.
2. They shoot their left neighbour, shoot their right neighbour, or deliberately miss.
3. If they shot, a random roll against their accuracy decides whether it hits.
4. A hit removes the target and the circle contracts.
5. Last player alive scores +1, everyone else 0.
6. If everybody keeps deliberately missing, the game is declared stuck and the +1 is split
   among the survivors.

## How to run it

From this folder:

```bash
../.venv/bin/jupyter lab step2_plain_game.ipynb
```

Read the **`.ipynb`**; the `.py` is the same content as plain text.

Takes about **5 seconds** to run everything, most of it the 100,000-episode cell.

## What you should see

One episode, printed step by step:

```
accuracies by position: [0.8 0.4 0.6]
circle order: [0, 1, 2]

t= 1  p0 (0.8) shoot_left at p2 (0.6)  -> HIT, p2 is out
t= 2  p0 (0.8) shoot_right at p1 (0.4)  -> missed
t= 3  p1 (0.4) shoot_right at p0 (0.8)  -> missed
...
t= 8  p0 (0.8) shoot_right at p1 (0.4)  -> HIT, p1 is out

ended because: sole_survivor
winner's accuracy: 0.8
```

Then the number this entire folder exists for — 100,000 episodes with everyone shooting their
strongest neighbour:

```
  accuracy 0.4  ->  survives 0.3705
  accuracy 0.6  ->  survives 0.3324
  accuracy 0.8  ->  survives 0.2971

stagnation rate: 0.0
survival probabilities add to: 1.0
```

And the check against the exact answer:

```
accuracy   exact      measured   difference   standard errors away
  0.8      0.2963     0.2971     +0.0008       +0.56
  0.6      0.3333     0.3324     -0.0010       -0.65
  0.4      0.3704     0.3705     +0.0002       +0.10
```

**Anything within about 3 standard errors is agreement, not a bug.** If a number is 10 or 20
standard errors out, the game is wrong.

Everyone deliberately missing, for contrast:

```
  stagnation rate: 1.0
  survival by accuracy: [0.3333 0.3333 0.3333]
Nobody wins.
```

Seeding:

```
identical for the same seed: True
different for a new seed  : True
```

## What to look at

**The weakest player survives most often — 0.3705 against the strongest player's 0.2971.**
Then look at the policy that produced it. It is four lines long and contains no cleverness
whatsoever. Survival-of-the-weakest is not something the weak player *does*; it is a
consequence of where everyone else points their gun. The two good shots are each other's
biggest threat, so they spend their turns on each other and nobody bothers with the 0.4 player.

**Two things both look like "nothing happened", and they are not the same.** A deliberate
`MISS` is a *choice*; firing and rolling badly is an *accident*. Only the deliberate one
advances the counter that detects a stuck game — a fired shot resets it whether or not it
landed, because the game is still moving. Get this backwards and the game still runs, it is
just a different game.

**A player sees only four numbers**: own accuracy, both neighbours' accuracies, and the
fraction still alive. Not the whole board. That restriction is what makes this policy directly
comparable to the trained neural agents later.

**Section 2.4 states the contracting-graph rule.** The step
function worked out the target twice — the second time *after* the victim had been removed —
so it was asking "who is my left neighbour" of a circle that had already changed shape. The
survival numbers were unaffected, but the printed trace claimed players were shooting people
who were already dead, and in one line had a player shooting itself. The lesson outlives the
one-line fix: on a contracting graph, **"my neighbour" is only true at one instant**.

## What it deliberately doesn't do

**No framework.** No `agent_selection`, no observation spaces, no standard reset signature.
That means no existing training code can drive it, which is exactly the problem step 3 solves.

**Only the `terminal` reward** — +1 to the winner, 0 to everyone else. Folder 03 adds a `shaped`
profile that pays a bonus for kills and a small amount for each step survived. It is not here,
because the exact targets are only valid for the terminal version.

**Next:** `step3-pettingzoo-env/` wraps this game in PettingZoo, and PettingZoo's own
conformance test then checks the wrapping.
