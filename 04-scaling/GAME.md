# The game, set up end to end

What a gruel is, mechanically, in the order the code does it. This is the reference for anyone reading a result in this folder and wondering what was actually played.

Everything below is the environment in `step1-train-any-n/gruel_env.py`, which is byte-identical to the copy every folder in deliverable 03 carries.

## 1. Choose the number of players

$N$. Deliverable 03 trains at 2, 3, 5 and 7. This folder trains at **10 and 15**.

## 2. Assign accuracies by formula

Each player gets a fixed probability of hitting what it aims at. They are evenly spaced and ascending:

$$p_k = 0.4 + 0.4 \cdot \frac{k}{N-1}, \qquad k = 0, 1, \dots, N-1$$

So $N=5$ gives `[0.4, 0.5, 0.6, 0.7, 0.8]`, and $N=3$ gives `[0.4, 0.6, 0.8]`.

An accuracy never changes during an episode. Nobody improves, nobody tires.

**The range is pinned to 0.4–0.8 whatever $N$ is, so a bigger field is also a tighter one.** The gap between neighbouring accuracies is 0.2000 at $N=3$, 0.0444 at $N=10$ and 0.0286 at $N=15$. A fifteen-player game is therefore not a three-player game with twelve more players in it — it is a game in which everyone is nearly equally skilled. Two things move together along the $N$ axis, and `step1-train-any-n/` reports what that does and does not permit.

## 3. Seat them on a circle

Each player takes a position and keeps it for the whole episode. **A player can only ever target its two immediate neighbours** — that is the graph constraint, and it is the whole difference between a gruel and a truel.

Two seating rules:

**`ranked`** — weakest at position 0, ascending round the circle, identical every episode. Because the circle wraps, the two best shots end up adjacent, so they shoot each other and the weakest is left alone. Note that the weakest's own left neighbour is the *strongest* player; the protection comes from where the strongest prefers to aim, not from distance.

**`random`** — a fresh shuffle on every `reset()`, so that protection becomes a lottery. The accuracies are the same multiset; only who sits where changes.

This folder uses `ranked`, so the three player counts that have an exact answer can still be read against it.

## 4. Choose a reward profile

| | win | eliminated | kill | survive a step |
|---|---|---|---|---|
| **`terminal`** | +1.0 | 0 | 0 | 0 |
| **`shaped`** | +1.0 | −1.0 | +0.1 | +0.001 |

`terminal` is the objective the mathematics solves: you are paid for winning and for nothing else. It is the only profile that can be graded against an exact answer, which is why every gate in the project uses it.

`shaped` adds hints to make learning easier, and the hints are not free. A player who kills several opponents and then dies can bank more than a player who quietly wins. The reward and the goal stop being the same thing.

## 5. Then play, one decision at a time

1. **Draw who acts** — uniformly at random from the players still alive, **with replacement**. This is not a rotation. The same player can act twice in a row, so one player may make eight decisions in an episode and another make one.
2. **That player observes four numbers** — its own accuracy, its left neighbour's, its right neighbour's, and the fraction of the field still alive. Four numbers whatever $N$ is. It is **not** told its own position, the full seating, or anything about players further round the circle.
3. **It chooses one of three actions** — shoot left, shoot right, or deliberately miss.
4. **Resolve the target before anyone is removed**, because a contraction changes who the neighbours are.
5. **Roll for the shot** — a uniform random draw, a hit if it falls below the shooter's accuracy. A shot that is *fired* resets the stuck-counter whether or not it lands; only a *deliberate* miss increases it.
6. **If it hit** — the shooter banks `kill`, the victim banks `eliminated`, both before the circle contracts, and the victim's position is dropped from the list of the living. **That removal is the contraction**: the two survivors on either side become adjacent because they become adjacent in the list.
7. **Everyone still standing banks `survive`** — zero under `terminal`.
8. **Check whether the episode is over.** If not, draw the next actor.

## 6. Three ways an episode ends

**`sole_survivor`** — one player left, who banks `win`. This is how almost every episode ends in practice.

**`stagnation`** — the deliberate-miss streak reaches $10 \times n_{\text{alive}}$. The win is then split among the survivors, $1/n_{\text{alive}}$ each. That split is what stops everyone abstaining forever being a free strategy: mutual abstention pays the same *in total* as a win, but less *each*.

**`hard_cap`** — $200N$ steps have passed. Same payout as stagnation, but flagged separately, because a run that keeps hitting this limit is a bug signal rather than a result.

The stuck-limit scales with the number alive, so it means the same thing late in a fifteen-player game as it does in a three-player one.

## One real episode

$N=5$, `ranked` seating, `terminal` reward, every player shooting its stronger neighbour.

```
=== 1-3. setup: N=5, ranked seating, terminal reward ===
  accuracies by position: [0.4, 0.5, 0.6, 0.7, 0.8]
  circle order: [0, 1, 2, 3, 4]
    position 0 (p=0.4)  left=4 (p=0.8)   right=1 (p=0.5)
    position 1 (p=0.5)  left=0 (p=0.4)   right=2 (p=0.6)
    position 2 (p=0.6)  left=1 (p=0.5)   right=3 (p=0.7)
    position 3 (p=0.7)  left=2 (p=0.6)   right=4 (p=0.8)
    position 4 (p=0.8)  left=3 (p=0.7)   right=0 (p=0.4)
  reward profile: {'win': 1.0, 'eliminated': 0.0, 'kill': 0.0, 'survive': 0.0}
  stagnation limit at full strength: 50   hard cap: 1000

=== 4-6. the loop: shoot the stronger neighbour, one episode ===
  step 1: actor player_4 (p=0.8)  sees [0.8 0.7 0.4 1. ]
          -> shoots position 3 (p=0.7)  HIT. circle [0, 1, 2, 3, 4] -> [0, 1, 2, 4]  (contracted)
  step 2: actor player_0 (p=0.4)  sees [0.4 0.8 0.5 0.8]
          -> shoots position 4 (p=0.8)  missed. circle unchanged, miss_streak=0
  step 3: actor player_4 (p=0.8)  sees [0.8 0.6 0.4 0.8]
          -> shoots position 2 (p=0.6)  HIT. circle [0, 1, 2, 4] -> [0, 1, 4]  (contracted)
  step 4: actor player_1 (p=0.5)  sees [0.5 0.4 0.8 0.6]
          -> shoots position 4 (p=0.8)  HIT. circle [0, 1, 4] -> [0, 1]  (contracted)
  step 5: actor player_1 (p=0.5)  sees [0.5 0.4 0.4 0.4]
          -> shoots position 0 (p=0.4)  HIT. circle [0, 1] -> [1]  (contracted)

  ended: reason=sole_survivor  after 5 steps
  episode totals: {'player_0': 0.0, 'player_1': 1.0, 'player_2': 0.0, 'player_3': 0.0, 'player_4': 0.0}
  elimination order (positions): [3, 2, 4, 0]
```

## Four things in that trace worth seeing

**`player_4` acts at step 1 and again at step 3.** The draw is with replacement, so turn order is not a queue and cannot be reasoned about as one.

**At step 3 the strongest shoots position 2, not position 3.** Position 3 is already dead and the circle has contracted, so its stronger neighbour is now a different player. This is why the code resolves the target before removing anyone.

**At step 5 `player_1` sees `[0.5 0.4 0.4 0.4]`** — its left and right are both position 0, because with two players alive both neighbours are the same opponent. There is no choice of target to observe, which is exactly why the targeting measurement throws these decisions away rather than counting them.

**The 0.8 player made both early kills and then died; the 0.5 player won** having made two of the five decisions. One episode proves nothing, but it is the shape of the entire project's result: shooting well attracts return fire, because the strongest neighbour is the one everybody else wants gone.

## Where the numbers in this folder come from

A run is one `(N, seating, reward)` combination trained from scratch at a given seed, then measured over fresh episodes with the policy still sampling rather than taking its best action. Each step's README states the budget it used and what it measured. The settings above are the only knobs; everything else in a result follows from them.
