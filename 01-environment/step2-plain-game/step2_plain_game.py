# %% [markdown]
# # Step 2 — The whole game, with no framework at all
#
# Step 1 gave us the shape. This notebook adds the **rules**, in plain Python. No
# Gymnasium, no PettingZoo, no libraries beyond numpy. Just a class with the game in it.
#
# I am writing it this way first on purpose. When we wrap the same game in PettingZoo in
# step 3, the only honest way to understand what the framework is *for* is to already know
# what it is wrapping.
#
# The rules, in full:
#
# 1. One player is chosen **at random** to act. Not in turn order — at random, every step,
#    and the same player can be chosen twice in a row.
# 2. They choose one of three actions: shoot left, shoot right, or deliberately miss.
# 3. If they shoot, a random roll against their accuracy decides whether it hits.
# 4. A hit removes the target, and the circle contracts.
# 5. Last player standing wins and scores +1. Everyone else scores 0.
# 6. If everybody keeps deliberately missing, the game is declared stuck and the reward is
#    split between the survivors.

# %%
N = 3
SEED = 0

import numpy as np

# The three actions. Using names instead of the bare numbers 0, 1, 2 everywhere.
SHOOT_LEFT = 0
SHOOT_RIGHT = 1
MISS = 2

ACTION_NAMES = {SHOOT_LEFT: "shoot_left", SHOOT_RIGHT: "shoot_right", MISS: "miss"}

# %% [markdown]
# ## 2.1 Two rules that are easy to get wrong
#
# Both of these are worth reading before the code, because they are quiet. Get them wrong
# and the game still runs, it just isn't the right game any more.
#
# **The actor is drawn with replacement.** Every step, pick uniformly at random from the
# players still alive. This is *not* "everyone takes a turn". With 3 players alive, the
# same player acting three times in a row has probability 1/27, and that is allowed. This is
# what "random order" means in the literature, and the analytical answer 8/27, 9/27, 10/27
# only holds for this version of the game.
#
# **A shot that misses is not the same as a deliberate miss.** There are two completely
# different things that both look like "nothing happened":
#
# | | What it is | Does it reset the stuck-counter? |
# |---|---|---|
# | `MISS` action | A **choice**. The player decided not to shoot | No — it increases it |
# | Fired and rolled badly | An **accident**. The player tried and failed | Yes — a shot was fired |
#
# The counter that detects a stuck game counts *deliberate* misses in a row. Firing and
# missing resets it, because the game is still moving.

# %% [markdown]
# ## 2.2 The game

# %%
class Gruel:
    """An N-player elimination duel on a contracting circle."""

    def __init__(self, n=3, seating="random", seed=0):
        self.n = n
        self.seating = seating
        self.rng = np.random.default_rng(seed)

        # Stop a stuck game: 10 deliberate misses in a row per living player.
        self.stagnation_rounds = 10
        # An absolute limit on steps, so nothing can run forever.
        self.max_steps = 200 * n

    # ---- setting up a new episode ----

    def reset(self):
        # Accuracies, evenly spaced, smallest first.
        accuracies_ascending = []
        for k in range(self.n):
            fraction = k / (self.n - 1)
            value = 0.4 + (0.8 - 0.4) * fraction
            accuracies_ascending.append(value)
        accuracies_ascending = np.array(accuracies_ascending)

        # Place them onto the circle positions.
        if self.seating == "ranked":
            self.accuracies = accuracies_ascending.copy()
        else:
            self.accuracies = self.rng.permutation(accuracies_ascending)

        self.order = list(range(self.n))   # who is alive, in circular order
        self.t = 0                         # how many steps have happened
        self.miss_streak = 0               # deliberate misses in a row
        self.elimination_order = []        # who died, in order
        self.finished = False
        self.reason = None
        self.rewards = np.zeros(self.n)

        # Choose who acts first.
        self.actor = self.draw_actor()

    # ---- small helpers ----

    def n_alive(self):
        return len(self.order)

    def draw_actor(self):
        """Pick a living player uniformly at random. With replacement, every step."""
        return int(self.rng.choice(self.order))

    def neighbours(self, position):
        index_in_order = self.order.index(position)
        how_many_alive = len(self.order)
        left = self.order[(index_in_order - 1) % how_many_alive]
        right = self.order[(index_in_order + 1) % how_many_alive]
        return left, right

    def stagnation_threshold(self):
        """The stuck-limit scales down as players die."""
        return self.stagnation_rounds * self.n_alive()

    def observe(self, position):
        """What a player is allowed to know: four numbers, and that is all."""
        left, right = self.neighbours(position)
        my_accuracy = self.accuracies[position]
        left_accuracy = self.accuracies[left]
        right_accuracy = self.accuracies[right]
        fraction_alive = self.n_alive() / self.n
        return np.array([my_accuracy, left_accuracy, right_accuracy, fraction_alive])

    def target_of(self, position, action):
        """Which position an action points at. None for a deliberate miss."""
        if action == MISS:
            return None
        left, right = self.neighbours(position)
        if action == SHOOT_LEFT:
            return left
        else:
            return right

    # ---- one step of the game ----

    def step(self, action):
        """The acting player does `action`. Returns a small dict describing what happened."""
        shooter = self.actor
        self.t = self.t + 1

        target = None
        victim = None
        hit = False

        if action == MISS:
            # A deliberate miss. Nothing happens, and the stuck-counter goes up.
            self.miss_streak = self.miss_streak + 1
        else:
            # Work out who is being shot at BEFORE anybody is removed. Once the
            # circle contracts, "my left neighbour" means somebody else.
            target = self.target_of(shooter, action)

            # A shot was fired, so the game is moving. Reset the stuck-counter
            # whether or not the shot actually lands.
            self.miss_streak = 0

            # Roll against the shooter's accuracy.
            roll = self.rng.random()
            hit = roll < self.accuracies[shooter]

            if hit:
                victim = target

        if victim is not None:
            self.order.remove(victim)
            self.elimination_order.append(victim)

        self.check_if_finished()

        if not self.finished:
            self.actor = self.draw_actor()

        return {
            "t": self.t,
            "shooter": shooter,
            "action": action,
            "target": target,
            "hit": hit,
            "victim": victim,
            "n_alive": self.n_alive(),
        }

    def check_if_finished(self):
        """Three ways an episode can end."""
        if self.n_alive() == 1:
            # Somebody won.
            self.finished = True
            self.reason = "sole_survivor"
            winner = self.order[0]
            self.rewards[winner] = 1.0

        elif self.miss_streak >= self.stagnation_threshold():
            # Everyone kept deliberately missing. Split the prize.
            self.finished = True
            self.reason = "stagnation"
            share = 1.0 / self.n_alive()
            for position in self.order:
                self.rewards[position] = share

        elif self.t >= self.max_steps:
            # Ran out of steps. Should basically never happen; if it does, something
            # is wrong rather than interesting.
            self.finished = True
            self.reason = "hard_cap"
            share = 1.0 / self.n_alive()
            for position in self.order:
                self.rewards[position] = share


# %% [markdown]
# ## 2.3 A policy to play it with
#
# The game needs somebody to choose actions. The simplest sensible choice is **shoot the
# stronger of your two neighbours** — the biggest threat is the best shot who can reach you.
#
# Note what it looks at: only `observation`, the four numbers. It does not get to see the
# whole board. This is the same restriction the trained neural agents will have, which is
# what makes them comparable.

# %%
def shoot_strongest(observation, rng):
    """Shoot whichever neighbour has the higher accuracy."""
    left_accuracy = observation[1]
    right_accuracy = observation[2]

    if left_accuracy > right_accuracy:
        return SHOOT_LEFT
    else:
        return SHOOT_RIGHT


# %% [markdown]
# ## 2.4 One episode, printed step by step

# %%
game = Gruel(n=N, seating="random", seed=SEED)
game.reset()

print("accuracies by position:", np.round(game.accuracies, 2))
print("circle order:", game.order)
print()

while not game.finished:
    position = game.actor
    observation = game.observe(position)
    action = shoot_strongest(observation, game.rng)
    result = game.step(action)

    shooter = result["shooter"]
    shooter_accuracy = game.accuracies[shooter]

    line = f"t={result['t']:>2}  p{shooter} ({shooter_accuracy:.1f}) {ACTION_NAMES[action]}"
    if result["action"] != MISS:
        target = result["target"]
        target_accuracy = game.accuracies[target]
        line = line + f" at p{target} ({target_accuracy:.1f})"
        if result["hit"]:
            line = line + "  -> HIT, p" + str(target) + " is out"
        else:
            line = line + "  -> missed"
    print(line)

print()
print("ended because:", game.reason)
print("survivor:", game.order)
print("rewards by position:", game.rewards)
print("winner's accuracy:", game.accuracies[game.order[0]])

# %% [markdown]
# ### One rule for contracting graphs
#
# **"My neighbour" is only true at one instant.** Work out the target *before* removing
# anyone: after `self.order.remove(victim)` the circle has already contracted, so the same
# neighbour lookup returns a different player. Any code that reads a neighbour has to be
# clear about whether it means before or after the contraction.

# %% [markdown]
# ## 2.5 Now run it many times
#
# One episode tells you nothing — it is a coin flip. The question the project actually asks
# is what happens **on average**, and specifically: does the *weakest* player win most
# often?
#
# Careful with the bookkeeping here. We want survival by **accuracy**, not by position,
# because seating is reshuffled each episode so position means nothing across episodes.

# %%
def run_many(n, episodes, policy, seed):
    """Play `episodes` games and count how often each accuracy level survives."""
    game = Gruel(n=n, seating="random", seed=seed)

    # The accuracy values that exist, smallest first. Same formula as always.
    accuracy_levels = []
    for k in range(n):
        fraction = k / (n - 1)
        accuracy_levels.append(0.4 + (0.8 - 0.4) * fraction)
    accuracy_levels = np.array(accuracy_levels)

    wins_per_level = np.zeros(n)
    stagnations = 0

    for episode in range(episodes):
        game.reset()

        while not game.finished:
            position = game.actor
            observation = game.observe(position)
            action = policy(observation, game.rng)
            game.step(action)

        if game.reason != "sole_survivor":
            stagnations = stagnations + 1

        # Credit the reward to the accuracy level that earned it.
        for position in range(n):
            if game.rewards[position] > 0:
                accuracy_here = game.accuracies[position]
                # Find which level this accuracy is.
                differences = np.abs(accuracy_levels - accuracy_here)
                level_index = int(np.argmin(differences))
                wins_per_level[level_index] += game.rewards[position]

    survival = wins_per_level / episodes
    stagnation_rate = stagnations / episodes
    return accuracy_levels, survival, stagnation_rate


EPISODES = 100_000

levels, survival, stagnation_rate = run_many(N, EPISODES, shoot_strongest, SEED)

print(f"{EPISODES:,} episodes, N={N}, everyone shooting their strongest neighbour")
print()
for i in range(len(levels)):
    print(f"  accuracy {levels[i]:.1f}  ->  survives {survival[i]:.4f}")
print()
print("stagnation rate:", stagnation_rate)
print("survival probabilities add to:", survival.sum())

# %% [markdown]
# ## 2.6 The number that matters
#
# At N=3 the exact answer is known. It is
#
# $$P(0.8) = \tfrac{8}{27} \approx 0.2963, \qquad
#   P(0.6) = \tfrac{9}{27} \approx 0.3333, \qquad
#   P(0.4) = \tfrac{10}{27} \approx 0.3704$$
#
# **The worst shot is the most likely to survive.** Not because it does anything clever —
# look at the policy, it has no cleverness in it at all — but because the two good shots
# are busy shooting each other while nobody prioritises the weak one.
#
# Folder 02 derives those fractions properly. For now, just check the simulation lands on
# them.

# %%
if N == 3:
    exact = np.array([8 / 27, 9 / 27, 10 / 27])   # for accuracies 0.8, 0.6, 0.4

    # `survival` is ordered weakest-first, `exact` is ordered strongest-first.
    measured_strongest_first = survival[::-1]

    # Standard error of a proportion from `EPISODES` samples.
    standard_error = np.sqrt(exact * (1 - exact) / EPISODES)

    print("accuracy   exact      measured   difference   standard errors away")
    accuracies_strongest_first = levels[::-1]
    for i in range(3):
        difference = measured_strongest_first[i] - exact[i]
        how_many_se = difference / standard_error[i]
        print(f"  {accuracies_strongest_first[i]:.1f}      "
              f"{exact[i]:.4f}     {measured_strongest_first[i]:.4f}     "
              f"{difference:+.4f}       {how_many_se:+.2f}")

    print()
    print("Anything within about 3 standard errors is agreement, not a bug.")

# %% [markdown]
# ## 2.7 What deliberate missing does
#
# The `MISS` action exists, and the original project proposal expected the weakest player
# to use it — the old idea of "firing in the air" to avoid making yourself a target.
#
# It turns out that is wrong for *this* version of the game, and it is worth seeing why
# with your own eyes. If everybody always misses on purpose, nobody ever dies.

# %%
def always_miss(observation, rng):
    return MISS


levels_m, survival_m, stagnation_m = run_many(N, 2_000, always_miss, SEED)

print("Everyone deliberately missing, 2,000 episodes:")
print("  stagnation rate:", stagnation_m)
print("  survival by accuracy:", np.round(survival_m, 4))
print()
print(f"Every episode ends stuck, and all {N} players 'survive' with 1/{N} each.")
print("Nobody wins. This is the reference point for 'the agents learned nothing'.")

# %% [markdown]
# ## 2.8 The seeding check

# %%
levels_a, survival_a, _ = run_many(N, 5_000, shoot_strongest, SEED)
levels_b, survival_b, _ = run_many(N, 5_000, shoot_strongest, SEED)
levels_c, survival_c, _ = run_many(N, 5_000, shoot_strongest, SEED + 1)

print("seed 0, first run :", np.round(survival_a, 5))
print("seed 0, second run:", np.round(survival_b, 5))
print("seed 1            :", np.round(survival_c, 5))
print()
print("identical for the same seed:", np.array_equal(survival_a, survival_b))
print("different for a new seed  :", not np.array_equal(survival_a, survival_c))

# %% [markdown]
# ## What to take from step 2
#
# - The actor is drawn **at random with replacement** each step. Not turn order.
# - A deliberate `MISS` and a fired shot that misses are different events, and only the
#   first one advances the stuck-counter.
# - A player sees only four numbers: own accuracy, both neighbours' accuracies, and the
#   fraction still alive.
# - Survival is counted **by accuracy**, never by position, because seating is reshuffled.
# - With a policy containing no strategy at all, the weakest player still survives most
#   often — about 0.370 at N=3.
#
# Next: `step3_pettingzoo_env.py` wraps this exact game in PettingZoo, so we can see what
# the framework adds and what it demands in return.
