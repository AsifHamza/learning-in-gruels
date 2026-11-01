# %% [markdown]
# # Step 1 — The circle, and what "contraction" means
#
# This is the smallest piece of the whole project: the **shape** of the game.
#
# N players sit on a circle. Each one has two neighbours, a left and a right, and those
# are the only two people they are allowed to shoot. When somebody dies, they are removed
# and their two former neighbours become neighbours of each other. The circle gets
# smaller. That is **contraction**, and it is the one thing that makes this game different
# from an ordinary free-for-all where anyone can shoot anyone.
#
# Nothing here is random except the seating, and nothing here is machine learning. By the
# end of this notebook you should be able to answer: who can shoot whom, and how does that
# change when someone dies?

# %%
# The two settings for this whole notebook. Change N and re-run everything.
N = 3
SEED = 0

import numpy as np

# %% [markdown]
# ## 1.1 The accuracies
#
# Every player has an accuracy: the probability that a shot they fire hits. The project
# always spreads them evenly between 0.4 and 0.8.
#
# The formula is
#
# $$p_k = \text{lo} + (\text{hi} - \text{lo}) \cdot \frac{k}{n-1}, \qquad k = 0, 1, \dots, n-1$$
#
# At N=3 that gives exactly 0.4, 0.6, 0.8 — the three numbers the whole project is built
# around.

# %%
def accuracy_set(n, lo=0.4, hi=0.8):
    """The n accuracy values, evenly spaced from lo to hi, smallest first."""
    accuracies = []
    for k in range(n):
        # k / (n - 1) goes 0, ..., 1 as k goes 0, ..., n-1.
        fraction = k / (n - 1)
        value = lo + (hi - lo) * fraction
        accuracies.append(value)
    return np.array(accuracies)


accuracies_ascending = accuracy_set(N)
print("accuracies, weakest first:", accuracies_ascending)

# %% [markdown]
# ## 1.2 Seating
#
# The accuracies now have to be placed *onto* the circle. There are two ways:
#
# - **ranked** — put them in order around the circle, so the strongest sits next to the
#   second strongest.
# - **random** — shuffle them, so a strong player might sit between two weak ones.
#
# The project uses `random` by default, and re-shuffles every single episode. That matters
# more than it looks: it means a result like "the weakest survives 37% of the time" is a
# statement about weak players *in general*, not about one particular arrangement.
#
# Note carefully what gets shuffled. **Position 0 is always position 0.** What moves is
# which accuracy is sitting there.

# %%
def seat_accuracies(accuracies_ascending, seating, rng):
    """Decide which accuracy sits at which position on the circle."""
    if seating == "ranked":
        return accuracies_ascending.copy()
    if seating == "random":
        return rng.permutation(accuracies_ascending)
    raise ValueError("seating must be 'ranked' or 'random'")


rng = np.random.default_rng(SEED)

seated_ranked = seat_accuracies(accuracies_ascending, "ranked", rng)
seated_random = seat_accuracies(accuracies_ascending, "random", rng)

print("ranked seating :", seated_ranked)
print("random seating :", seated_random)
print()
print("Read those as: position 0 has accuracy", seated_random[0])

# %% [markdown]
# ## 1.3 The circle itself
#
# Here is the idea that makes this simple. You might expect a graph class with a list of
# edges, and code to delete edges and add new ones when a player dies.
#
# You don't need any of that. Keep a **list of positions in circular order**. Your
# neighbours are whoever is next to you *in the list*. Then to kill someone, remove them
# from the list — and their two neighbours are now automatically next to each other.
#
# Contraction is not something the code does. It is something that happens because of how
# the data is stored.

# %%
class CircleGraph:
    """The living players, kept in circular order."""

    def __init__(self, n):
        self.n = n
        # At the start, everyone is alive and sitting in order 0, 1, 2, ...
        self.order = list(range(n))

    def n_alive(self):
        return len(self.order)

    def is_alive(self, position):
        return position in self.order

    def neighbours(self, position):
        """The (left, right) neighbours of a position, going around the circle."""
        index_in_order = self.order.index(position)
        how_many_alive = len(self.order)

        # The % wraps around: the person before index 0 is the last person in the list.
        left_index = (index_in_order - 1) % how_many_alive
        right_index = (index_in_order + 1) % how_many_alive

        left = self.order[left_index]
        right = self.order[right_index]
        return left, right

    def eliminate(self, position):
        """Remove a player. Returns the two neighbours who are now joined up."""
        left, right = self.neighbours(position)
        self.order.remove(position)
        return left, right

    def adjacency(self):
        """The n by n table: 1 means 'these two are neighbours', 0 means they are not."""
        table = np.zeros((self.n, self.n), dtype=int)
        how_many_alive = len(self.order)

        # With only 1 alive there are no pairs. With exactly 2 alive they are each
        # other's neighbour on both sides, which would be one edge, not two.
        if how_many_alive < 2:
            return table

        for index_in_order in range(how_many_alive):
            this_player = self.order[index_in_order]
            next_index = (index_in_order + 1) % how_many_alive
            next_player = self.order[next_index]
            table[this_player][next_player] = 1
            table[next_player][this_player] = 1

        return table


# %% [markdown]
# ## 1.4 Who can shoot whom, at the start

# %%
graph = CircleGraph(N)

print("alive, in circular order:", graph.order)
print()
for position in graph.order:
    left, right = graph.neighbours(position)
    print(f"position {position} (accuracy {seated_random[position]:.1f})"
          f" can shoot left={left} or right={right}")

# %% [markdown]
# At N=3 something special happens: everyone can shoot everyone. A triangle is both a
# circle *and* a complete graph. This is why N=3 is the right place to start the project —
# the graph constraint is switched off, so nothing is being tested except the game itself.
#
# The constraint only starts to matter at N=4 and above, where you have opponents you
# cannot reach.

# %%
# Proof of that claim: at N=3 every player has 2 neighbours out of 2 possible opponents.
number_of_opponents = N - 1
left, right = graph.neighbours(0)
number_of_reachable = len({left, right})

print(f"N = {N}")
print(f"opponents:            {number_of_opponents}")
print(f"reachable neighbours: {number_of_reachable}")
if number_of_reachable == number_of_opponents:
    print("=> everyone can reach everyone. The graph is not restricting anything.")
else:
    print("=> some opponents cannot be reached. The graph is doing real work.")

# %% [markdown]
# ## 1.5 Watching a contraction happen
#
# Now the important bit. Let's kill somebody and look at the circle before and after.
#
# To see anything interesting we need at least 4 players, so this section builds its own
# bigger circle regardless of what `N` is set to.

# %%
demo = CircleGraph(6)

print("BEFORE")
print("  order:", demo.order)
print("  neighbours of 1:", demo.neighbours(1))
print("  neighbours of 3:", demo.neighbours(3))
print("  are 1 and 3 neighbours?", demo.adjacency()[1][3] == 1)

victim = 2
new_left, new_right = demo.eliminate(victim)

print()
print(f"player {victim} has been eliminated")
print()
print("AFTER")
print("  order:", demo.order)
print("  neighbours of 1:", demo.neighbours(1))
print("  neighbours of 3:", demo.neighbours(3))
print("  are 1 and 3 neighbours?", demo.adjacency()[1][3] == 1)
print()
print(f"  the contraction joined {new_left} and {new_right}")

# %% [markdown]
# Player 1 and player 3 were strangers. Player 2 died and now they can shoot each other.
# No edge-editing code ran — `self.order.remove(2)` did all of it.
#
# This is the mechanism the entire thesis rests on. A player's set of possible targets
# changes over the course of an episode, which means who you *should* shoot depends on who
# you will be left facing afterwards.

# %% [markdown]
# ## 1.6 The circle stays a circle
#
# One thing that could go wrong: contraction could break the shape. If it did, the game
# would quietly stop being the game we think we are studying.
#
# The test is that every living player has exactly **2** neighbours, all the way down to 3
# players remaining. (At 2 players there is only one edge between them, so each has 1.)

# %%
check = CircleGraph(8)
kill_order = [5, 0, 3, 7, 1, 2]

for victim in kill_order:
    check.eliminate(victim)
    table = check.adjacency()

    # Count each living player's neighbours by adding up their row of the table.
    degrees = []
    for position in check.order:
        row = table[position]
        degrees.append(int(row.sum()))

    alive_now = check.n_alive()
    print(f"killed {victim} -> {alive_now} alive, order {check.order}, degrees {degrees}")

    if alive_now >= 3:
        for degree in degrees:
            assert degree == 2, "the circle broke"

print()
print("Every living player kept exactly 2 neighbours. Still a circle.")

# %% [markdown]
# ## 1.7 The seeding check
#
# The only randomness in this notebook is the seating. Same seed must mean same seating,
# every time, or nothing later in the project is reproducible.

# %%
first_rng = np.random.default_rng(SEED)
second_rng = np.random.default_rng(SEED)

first_seating = seat_accuracies(accuracies_ascending, "random", first_rng)
second_seating = seat_accuracies(accuracies_ascending, "random", second_rng)

different_rng = np.random.default_rng(SEED + 1)
different_seating = seat_accuracies(accuracies_ascending, "random", different_rng)

print("seed 0, first time :", first_seating)
print("seed 0, second time:", second_seating)
print("seed 1             :", different_seating)
print()
print("same seed gives the same seating:", np.array_equal(first_seating, second_seating))

# %% [markdown]
# Notice that `rng` is an **object** that gets passed in, not a global setting. Every time
# you draw from it, it moves on. That is why `seated_ranked` and `seated_random` back in
# section 1.2 came from the *same* generator and are still different draws.
#
# The alternative — `np.random.seed(0)` — sets one hidden global state that anything in the
# program can quietly consume. Passing the object makes it visible who used randomness.

# %% [markdown]
# ## Reproducibility: run the same thing twice
#
# This notebook shuffles, so it has to prove the shuffle is repeatable. Two fresh
# generators with the same seed must produce the same seatings — not similar ones,
# identical ones.

# %%
def ten_seatings(seed):
    generator = np.random.default_rng(seed)
    seatings = []
    for episode in range(10):
        seated = seat_accuracies(accuracy_set(5), "random", generator)
        seatings.append([round(a, 4) for a in seated])
    return seatings


first_run = ten_seatings(SEED)
second_run = ten_seatings(SEED)

print("first seating, run 1:", first_run[0])
print("first seating, run 2:", second_run[0])
print()
print("all ten identical for the same seed:", first_run == second_run)

# A different seed must give something different, or the seed is being ignored.
different = ten_seatings(SEED + 1)
print("a different seed gives different seatings:", first_run != different)

# %% [markdown]
# ## What to take from step 1
#
# - Accuracies are evenly spaced on [0.4, 0.8]; at N=3 they are exactly 0.4, 0.6, 0.8.
# - Seating decides which accuracy sits at which position, and is re-shuffled each episode.
# - The circle is just a **list**. Contraction is `list.remove()`.
# - At N=3 the circle is a complete graph, so the adjacency constraint does nothing. It
#   starts doing real work at N=4 and above.
#
# Next: `step2_plain_game.py` puts a game on top of this shape — turns, shooting, dying,
# winning.
