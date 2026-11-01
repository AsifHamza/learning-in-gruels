# %% [markdown]
# # Step 3 — Five players on a circle, where the graph finally matters
#
# At N=3 the circle is a triangle, so everyone can shoot everyone and the graph does nothing.
# At N=5 that stops being true: you have four opponents and can only reach **two** of them.
#
# This notebook produces the N=5 targets — the numbers the trained agents in folder 03 will
# have to hit. They are worth being careful about for one reason: **they are not in any
# paper.** At N=3 we could check against Kilgour & Brams. Here there is nothing to check
# against, so the only protection is that the same code got all six of their rows right.
#
# It also produces two results that are more interesting than the numbers themselves, and one
# genuine open problem that the project has not resolved.

# %%
import itertools

ABSTAIN = None

LOW = 0.4
HIGH = 0.8
N = 5

# %% [markdown]
# ## 3.1 What changes from N=3
#
# Three things, and only the first is obvious.
#
# **You can only shoot your neighbours.** With 5 players you have 4 opponents and 2 of them
# are out of reach. So "shoot the strongest" now means "shoot the stronger of the two you can
# actually hit", which may not be the strongest player in the game at all.
#
# **The equilibrium has to be worked out state by state.** At N=3 there was one interesting
# state. At N=5 there are many, and what is best to do now depends on what the equilibrium
# will be *after* somebody dies. So we have to solve the small states first and work upwards.
#
# **Seating matters.** At N=3 every arrangement of accuracies around a triangle is the same
# game. On a 5-circle it is not: sitting between the two strongest players is a different
# proposition from sitting between the two weakest. This turns out to matter more than I
# expected, and section 3.8 is about it.

# %%
def accuracy_set(n, lo=LOW, hi=HIGH):
    """Evenly spaced accuracies, smallest first."""
    values = []
    for k in range(n):
        fraction = k / (n - 1)
        values.append(lo + (hi - lo) * fraction)
    return values


print("N=5 accuracies:", accuracy_set(5))
print()
print("Ranked seating puts them round the circle in that order,")
print("so position 0 is the weakest and position 4 the strongest,")
print("and position 4 sits next to position 0 because it is a circle.")

# %% [markdown]
# ## 3.2 How many states are there?
#
# A state is just "who is still alive". With 5 players that is $2^5 = 32$ subsets. Subtract
# the 5 states with one player alive (the game is over) and the 1 empty state, leaving **26**
# states where a decision has to be made.
#
# This number matters beyond bookkeeping: it is what makes exact solving possible at all. At
# N=15 there would be $2^{15} - 16 = 32{,}752$ states, and each needs every combination of
# strategies checked, which is a great deal of work. Where the method actually stops being usable
# is a separate question that step 4 settles by running it, and the answer is not cost.

# %%
def all_states(n):
    """Every set of alive players with at least 2 in it, smallest sets first."""
    states = []
    for size in range(2, n + 1):
        for combination in itertools.combinations(range(n), size):
            states.append(combination)
    return states


states = all_states(5)
print("states needing a decision:", len(states))
print()
for size in range(2, 6):
    count = 0
    for state in states:
        if len(state) == size:
            count = count + 1
    print(f"  {size} players alive: {count} states")

print()
print("For comparison:")
for n in [3, 5, 10, 15]:
    print(f"  N={n:>2}: {2**n - n - 1:>6,} states")

# %% [markdown]
# ## 3.3 Who can shoot whom on a circle
#
# The clean part of contraction: because removing a player from the circular order
# automatically joins their neighbours, **the set of survivors alone tells you the whole
# graph.** You never have to remember how you got to a state. That is what makes it safe to
# store answers in a dictionary keyed by the alive-set.

# %%
def neighbours_on_circle(alive, position):
    """The players `position` can shoot. Just one of them if only two are left."""
    how_many = len(alive)
    index_here = alive.index(position)

    left = alive[(index_here - 1) % how_many]
    right = alive[(index_here + 1) % how_many]

    if left == right:
        return [left]
    return [left, right]


alive = (0, 1, 2, 3, 4)
print("full circle:")
for position in alive:
    print(f"  p{position} can shoot {neighbours_on_circle(alive, position)}")

alive_after = (0, 2, 3, 4)
print()
print("after p1 dies:")
for position in alive_after:
    print(f"  p{position} can shoot {neighbours_on_circle(alive_after, position)}")
print()
print("p0 and p2 are now neighbours. They were not before.")

# %% [markdown]
# ## 3.4 The value of a state, given what everyone does
#
# Identical to step 2: add up the hazard aimed at each player, and average the smaller states
# weighted by those hazards.

# %%
def value_of_profile(accuracies, alive, choice_of, known_values):
    """Survival probability for everyone, when each living player shoots their choice."""
    n = len(accuracies)

    hazard = {}
    for shooter in alive:
        target = choice_of[shooter]
        if target is ABSTAIN:
            continue
        if target not in hazard:
            hazard[target] = 0.0
        hazard[target] = hazard[target] + accuracies[shooter]

    total_hazard = 0.0
    for target in hazard:
        total_hazard = total_hazard + hazard[target]

    # Nobody shooting: the game is stuck and the prize is shared.
    if total_hazard == 0.0:
        share = 1.0 / len(alive)
        result = [0.0] * n
        for player in alive:
            result[player] = share
        return result

    result = [0.0] * n
    for target in hazard:
        weight = hazard[target] / total_hazard

        remaining = []
        for player in alive:
            if player != target:
                remaining.append(player)
        smaller = known_values[tuple(remaining)]

        for player in range(n):
            result[player] = result[player] + weight * smaller[player]

    return result


# %% [markdown]
# ## 3.5 Finding the equilibrium in one state
#
# For a state, list every combination of choices, and keep the ones where no single player can
# improve their own survival by switching.
#
# The cost of this is worth noticing. Each living player has up to 3 options (two neighbours
# plus abstain), so a 5-player state needs $3^5 = 243$ combinations checked, and each check
# tests up to 10 deviations. That is fine at N=5 and hopeless at N=15.

# %%
def options_for(alive, shooter, topology="circle"):
    """Everything a player may do: shoot a reachable opponent, or abstain."""
    if topology == "circle":
        choices = list(neighbours_on_circle(alive, shooter))
    else:
        choices = []
        for other in alive:
            if other != shooter:
                choices.append(other)

    choices.append(ABSTAIN)
    return choices


def every_combination(alive, topology="circle"):
    """All ways the living players could simultaneously choose."""
    combinations = [{}]

    for shooter in alive:
        grown = []
        for partial in combinations:
            for choice in options_for(alive, shooter, topology):
                extended = dict(partial)
                extended[shooter] = choice
                grown.append(extended)
        combinations = grown

    return combinations


def find_equilibria(accuracies, alive, known_values, topology="circle"):
    """Every combination nobody wants to deviate from."""
    equilibria = []

    for combination in every_combination(alive, topology):
        values = value_of_profile(accuracies, alive, combination, known_values)
        is_stable = True

        for shooter in alive:
            for alternative in options_for(alive, shooter, topology):
                if alternative == combination[shooter]:
                    continue

                trial = dict(combination)
                trial[shooter] = alternative
                trial_values = value_of_profile(accuracies, alive, trial, known_values)

                if trial_values[shooter] > values[shooter] + 1e-12:
                    is_stable = False
                    break

            if not is_stable:
                break

        if is_stable:
            equilibria.append((combination, values))

    return equilibria


# %% [markdown]
# ## 3.6 Solving all 26 states, smallest first
#
# The order is the whole point. You cannot know what to do with 5 players alive until you know
# what happens with 4, and so on down to 2. So we start at the bottom and work up. This is
# **backward induction**.
#
# One wrinkle: a state can have more than one equilibrium, and then we have to choose. The
# project's rule is to take the one where the most shooting happens (highest total hazard),
# and to record how often a choice had to be made at all — because "there were several
# equilibria" is itself a result worth reporting.

# %%
def solve_all(accuracies, topology="circle"):
    """Backward induction over every state. Returns values, chosen profiles, and counts."""
    n = len(accuracies)
    known_values = {}
    chosen_profiles = {}
    how_many_equilibria = {}

    # Base cases: one player alive means that player has won.
    for player in range(n):
        values = [0.0] * n
        values[player] = 1.0
        known_values[(player,)] = values

    for size in range(2, n + 1):
        for alive in itertools.combinations(range(n), size):
            equilibria = find_equilibria(accuracies, alive, known_values, topology)

            if len(equilibria) == 0:
                raise RuntimeError(f"no equilibrium at state {alive}")

            how_many_equilibria[alive] = len(equilibria)

            # Choose the one with the most shooting.
            best_combination = None
            best_values = None
            best_hazard = -1.0

            for combination, values in equilibria:
                hazard_total = 0.0
                for shooter in alive:
                    if combination[shooter] is not ABSTAIN:
                        hazard_total = hazard_total + accuracies[shooter]

                if hazard_total > best_hazard:
                    best_hazard = hazard_total
                    best_combination = combination
                    best_values = values

            known_values[alive] = best_values
            chosen_profiles[alive] = best_combination

    everyone = tuple(range(n))
    return known_values[everyone], chosen_profiles, how_many_equilibria, known_values


# %% [markdown]
# ### First, prove it still gets N=3 right
#
# The generic circle solver should reproduce 8/27, 9/27, 10/27, because at N=3 the circle *is*
# the complete graph. If it does not, nothing below is trustworthy.

# %%
accuracies_n3 = accuracy_set(3)          # [0.4, 0.6, 0.8], weakest at position 0
survival_n3, profiles_n3, counts_n3, _ = solve_all(accuracies_n3, "circle")

print("accuracies by position:", accuracies_n3)
print()
print("position  accuracy   survival     exact       match")
exact_by_position = [10 / 27, 9 / 27, 8 / 27]
for position in range(3):
    difference = abs(survival_n3[position] - exact_by_position[position])
    print(f"   p{position}      {accuracies_n3[position]:.1f}      "
          f"{survival_n3[position]:.10f}  {exact_by_position[position]:.10f}  "
          f"{difference < 1e-12}")

print()
print("equilibria per state:", counts_n3)
print("sums to:", sum(survival_n3))

# %% [markdown]
# ## 3.7 Now N=5, ranked seating
#
# Ranked seating means accuracies go round the circle in order: position 0 is 0.4, position 4
# is 0.8, and because it is a circle those two sit next to each other.

# %%
accuracies_n5 = accuracy_set(5)
survival_n5, profiles_n5, counts_n5, all_values = solve_all(accuracies_n5, "circle")

print("N=5, circle, ranked seating")
print()
print("position  accuracy   survival")
for position in range(5):
    print(f"   p{position}      {accuracies_n5[position]:.1f}      {survival_n5[position]:.5f}")

print()
print("sums to:", sum(survival_n5))
print("states solved:", len(counts_n5))

# Did any state have more than one equilibrium?
multiple = []
for state in counts_n5:
    if counts_n5[state] > 1:
        multiple.append(state)

print("states with more than one equilibrium:", len(multiple))
if len(multiple) == 0:
    print("=> the equilibrium is unique everywhere, so no choice had to be made.")

# %% [markdown]
# ### Sorted by accuracy, which is how the project reports it

# %%
by_rank = []
for position in range(5):
    by_rank.append((accuracies_n5[position], survival_n5[position]))
by_rank.sort(reverse=True)

print("accuracy   survival     rank by survival")
ordered_survivals = sorted([value for _, value in by_rank], reverse=True)
for accuracy, value in by_rank:
    rank = ordered_survivals.index(value) + 1
    print(f"  {accuracy:.1f}      {value:.5f}      {rank} of 5")

# %% [markdown]
# ## 3.8 Two results worth stopping for
#
# **Survival is not monotone in accuracy.** The weakest player does best, as at N=3. But the
# strongest is only third, and the **second-strongest does worst of everybody**. That is not
# what "survival of the weakest" would lead you to expect — it is not a clean reversal of the
# ordering, it is a different shape entirely.
#
# The reason is the graph. On a circle you can only be shot by your two neighbours, so what
# matters is not how good you are but **who you are sitting next to**. Under ranked seating
# the second-strongest sits directly beside the strongest, which is the worst seat at the
# table.
#
# **Shoot-the-strongest is no longer the equilibrium.** At N=3 it was the unique one. Here it
# is not, and the difference is large.

# %%
def shoot_strongest_neighbour(alive, shooter, accuracies):
    """The heuristic from folder 01: fire at the better shot of your two neighbours."""
    reachable = neighbours_on_circle(alive, shooter)

    best_target = None
    best_accuracy = -1.0
    for candidate in reachable:
        if accuracies[candidate] > best_accuracy:
            best_accuracy = accuracies[candidate]
            best_target = candidate

    return best_target


def evaluate_fixed_policy(accuracies, policy, topology="circle"):
    """Exact survival under a policy that is fixed rather than solved for."""
    n = len(accuracies)
    known_values = {}

    for player in range(n):
        values = [0.0] * n
        values[player] = 1.0
        known_values[(player,)] = values

    for size in range(2, n + 1):
        for alive in itertools.combinations(range(n), size):
            choice_of = {}
            for shooter in alive:
                choice_of[shooter] = policy(alive, shooter, accuracies)
            known_values[alive] = value_of_profile(accuracies, alive, choice_of, known_values)

    return known_values[tuple(range(n))]


heuristic_survival = evaluate_fixed_policy(accuracies_n5, shoot_strongest_neighbour)

print("N=5 circle, ranked seating")
print()
print("accuracy   equilibrium   shoot-strongest   difference")
for position in range(5):
    equilibrium_value = survival_n5[position]
    heuristic_value = heuristic_survival[position]
    print(f"  {accuracies_n5[position]:.1f}       {equilibrium_value:.5f}       "
          f"{heuristic_value:.5f}         {heuristic_value - equilibrium_value:+.5f}")

print()
print("Both sum to 1:", round(sum(survival_n5), 10), round(sum(heuristic_survival), 10))

# %% [markdown]
# Those are two different games' worth of difference. So at N=5, **the trained agents must be
# compared against the equilibrium column, never against the heuristic** — which is easy to
# miss, because at N=3 the two coincide and it is easy to assume they always will.
#
# Let us confirm the heuristic really is not an equilibrium, by counting how many players
# would rather do something else.

# %%
# Build the heuristic's own continuation values, then look for profitable deviations.
known_under_heuristic = {}
for player in range(5):
    values = [0.0] * 5
    values[player] = 1.0
    known_under_heuristic[(player,)] = values

for size in range(2, 6):
    for alive in itertools.combinations(range(5), size):
        choice_of = {}
        for shooter in alive:
            choice_of[shooter] = shoot_strongest_neighbour(alive, shooter, accuracies_n5)
        known_under_heuristic[alive] = value_of_profile(
            accuracies_n5, alive, choice_of, known_under_heuristic)

deviations = 0
for size in range(2, 6):
    for alive in itertools.combinations(range(5), size):
        choice_of = {}
        for shooter in alive:
            choice_of[shooter] = shoot_strongest_neighbour(alive, shooter, accuracies_n5)
        base_values = value_of_profile(accuracies_n5, alive, choice_of, known_under_heuristic)

        for shooter in alive:
            for alternative in options_for(alive, shooter):
                if alternative == choice_of[shooter]:
                    continue
                trial = dict(choice_of)
                trial[shooter] = alternative
                trial_values = value_of_profile(
                    accuracies_n5, alive, trial, known_under_heuristic)
                if trial_values[shooter] > base_values[shooter] + 1e-12:
                    deviations = deviations + 1

print("profitable deviations from shoot-strongest, across all states:", deviations)
if deviations > 0:
    print("=> shoot-strongest is NOT an equilibrium at N=5.")

# %% [markdown]
# ## 3.9 The open problem: which seating?
#
# Everything above used **ranked** seating. But the environment in folder 01 reshuffles the
# accuracies every episode, which is **random** seating. So which numbers should the agents be
# held to?
#
# The obvious move is to solve every possible seating and average. There are $5! = 120$
# arrangements, which is nothing.

# %%
def seating_average(n):
    """Solve the circle for every arrangement of accuracies and average by accuracy value."""
    accuracies_ascending = accuracy_set(n)

    total_by_accuracy = {}
    count_by_accuracy = {}
    for accuracy in accuracies_ascending:
        total_by_accuracy[accuracy] = 0.0
        count_by_accuracy[accuracy] = 0

    how_many_seatings = 0

    for arrangement in itertools.permutations(accuracies_ascending):
        seated = list(arrangement)
        survival, _, _, _ = solve_all(seated, "circle")
        how_many_seatings = how_many_seatings + 1

        for position in range(n):
            accuracy_here = seated[position]
            total_by_accuracy[accuracy_here] += survival[position]
            count_by_accuracy[accuracy_here] += 1

    averaged = {}
    for accuracy in accuracies_ascending:
        averaged[accuracy] = total_by_accuracy[accuracy] / count_by_accuracy[accuracy]

    return averaged, how_many_seatings


averaged, seatings_done = seating_average(5)

print(f"averaged over all {seatings_done} seatings")
print()
print("accuracy   ranked seating   averaged over seatings")
for position in range(5):
    accuracy = accuracies_n5[position]
    print(f"  {accuracy:.1f}       {survival_n5[position]:.5f}          {averaged[accuracy]:.5f}")

total = 0.0
for accuracy in averaged:
    total = total + averaged[accuracy]
print()
print("averaged column sums to:", total)

# %% [markdown]
# ### And now the problem
#
# Look at the averaged column: it is **monotone**. The weakest does best, the strongest does
# worst, in order. The non-monotone shape — second-strongest doing worst — has vanished.
#
# So that striking result is a fact about **ranked seating**, not about the circle. The real
# project initially attributed it to the graph constraint and had to withdraw the claim.
#
# But the averaged column cannot simply be used as the target either, and this is the honest
# difficulty. Each of the 120 solves assumes the players **know** the arrangement — they are
# each computed as a complete-information game. Our agents do not know it. They see only
# `[own accuracy, left neighbour, right neighbour, fraction alive]`, so they cannot tell
# which of the 120 seatings they are in.
#
# That makes their situation a game of **incomplete information**, and the equilibrium of such
# a game is not in general the average of the equilibria of the games it might be. So:
#
# - **ranked seating** has a target we can defend, because there is nothing hidden.
# - **random seating** does not yet, because nobody has solved the incomplete-information
#   version.
#
# This is the project's main open question. It is not resolved here — this notebook exists to
# make the problem clear, not to solve it.

# %% [markdown]
# ## 3.10 How fast the work grows
#
# Counting the work: $2^N$ states, and in each one every player picks among up to 3 options, so
# up to $3^{N}$ combinations per state, each needing every deviation checked.
#
# This says how the cost scales. It does **not** say where the method stops — that needs the
# solver actually run upward until it fails, which is step 4's job.

# %%
print("     N     states    combinations per full state    rough total checks")
for n in [3, 5, 8, 10, 15]:
    states_count = 2 ** n - n - 1
    combinations_count = 3 ** n
    print(f"  {n:>4}   {states_count:>8,}    {combinations_count:>22,}    "
          f"{states_count * combinations_count:>18,.0f}")

print()
print("The growth is steep -- each extra player multiplies the work several times")
print("over -- and at N=15 the count is very large.")
print()
print("Steep growth does not by itself locate a boundary. Step 4 runs the solver")
print("upward until it fails, and finds the limit is not cost: at N=9 a state has")
print("no pure equilibrium, so backward induction over pure strategies has nothing")
print("to return.")

# %% [markdown]
# ## 3.11 How long this actually took

# %%
import time

start = time.time()
solve_all(accuracy_set(5), "circle")
print(f"one N=5 solve: {time.time() - start:.3f} seconds")

start = time.time()
solve_all(accuracy_set(6), "circle")
print(f"one N=6 solve: {time.time() - start:.3f} seconds")

start = time.time()
solve_all(accuracy_set(7), "circle")
print(f"one N=7 solve: {time.time() - start:.3f} seconds")

print()
print("Compare: folder 01 needed 100,000 random episodes for a worse answer at N=3.")
print("Exact is enormously cheaper than sampling, right up until it is impossible.")

# %% [markdown]
# ## 3.12 The seating-averaged column, written out in full
#
# Section 3.9 rounds the seating-averaged profile to five decimals. This prints it at twelve, one
# line per accuracy, in a fixed format, so any quotation of these numbers elsewhere can be
# checked against what this notebook computed.

# %%
for accuracy in accuracies_n5:
    print(f"seating-averaged N=5 accuracy {accuracy:.1f} survival {averaged[accuracy]:.12f}")

# %% [markdown]
# ## What to take from step 3
#
# - There are **26** states at N=5, and they are solved **smallest first** — backward
#   induction — because what you should do now depends on what happens after someone dies.
# - The alive-set alone determines the graph, which is what makes storing answers by
#   alive-set valid.
# - Under ranked seating the equilibrium is **unique in every state**, and survival is
#   **not monotone**: the weakest does best, but the second-strongest does worst.
# - That non-monotone shape is caused by **seating**, not by the graph. Averaged over all
#   seatings it disappears.
# - **Shoot-strongest is not the equilibrium at N=5**, unlike N=3. Agents must be compared
#   against the solved equilibrium.
# - The random-seating target is an **open problem**: agents cannot see the arrangement, so
#   their game is one of incomplete information, and averaging complete-information solutions
#   is not the same thing.
# - The work grows by roughly a factor of five per extra player, so cost rises fast. Where the
#   method actually stops is a different question, and step 4 answers it by running the solver
#   upward: the exact answers reach **N=8**, and N=9 fails because a reachable state has **no
#   pure equilibrium at all** rather than because it is slow.
#
# Step 4 takes that walk upward, and step 5 asks how you know the answers really are equilibria.
