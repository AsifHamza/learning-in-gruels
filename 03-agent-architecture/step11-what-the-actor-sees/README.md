# Step 11 — Can an actor that sees four numbers play the equilibrium?

**A check on what the actor can represent, at five and seven players, using the exact solver and the saved actors of steps 6 and 9. It trains nothing.**

## Why this step exists

The actor sees four numbers: its own accuracy, its two neighbours' accuracies, and the share of the field still alive. The equilibrium is computed from the whole set of survivors. Two states can therefore look the same to a player while the equilibrium asks for different actions in them. If that happens often, no actor of this kind can play the equilibrium exactly, and the seven-player failure of step 9 would have a cause that more training cannot remove. This step measures it.

## What it does

1. Solves the five- and seven-player games under ranked seating with folder 02's solver, copied into this folder. Every state has exactly one pure equilibrium, as in folder 02.
2. Builds, for every state with three or more alive and every player in it, the observation the environment would give that player, and checks all 454 against `GruelEnv.observe`.
3. Groups decisions by observation. A group **conflicts** if the equilibrium asks for more than one action within it. The **ceiling** is the most any four-number policy could agree with the equilibrium: in each group, the action with the most weight. Decisions are weighted as in folder 02 step 5: the probability of reaching the state under equilibrium play, times $1/H$.
4. Loads the saved actors (the five seeds of step 6 at five players, and the three seeds of step 9 at seven), checks each against the weight hash in its record, and reads from the network the probability it gives the equilibrium's action in every state. The **agreement** is that probability averaged with the same weights.

States with two players alive are left out, because both shooting actions then point at the same opponent.

## How to run it

```bash
../.venv/bin/jupyter lab step11_what_the_actor_sees.ipynb
```

It runs in a few seconds. Nothing is random.

## What you should see

**At five players no observation conflicts.** The ceiling is 1.0000, and the trained actors agree with the equilibrium on 0.9888 of decisions on average (0.9858 to 0.9922 across the five seeds).

**At seven players two observations conflict**, both with six players alive and both in the upper middle of the field. Together they carry 0.0454 of decisions. The ceiling is therefore 0.9871: a four-number policy cannot play the seven-player equilibrium exactly, but it can come close.

**The trained seven-player actors fall well short of that ceiling.** They agree on 0.9144 of decisions on average (0.8810 to 0.9384), 0.9264 of the ceiling. By players alive, the agreement is 0.8571 with all seven alive and 0.8385 with three alive, where nothing conflicts and the ceiling is 1.0000. With six alive, where the conflicts are, it is 0.8816 against a ceiling of 0.9347.

**So the observation limit is real but small, and it is not what makes the seven-player agent fail.** Most of the disagreement is in states a four-number actor could play correctly.

Step 9's five-player seeds 0 to 2 are the same configuration as step 6's seeds 0 to 2, and the notebook confirms their saved weights are identical, so they are not counted twice.
