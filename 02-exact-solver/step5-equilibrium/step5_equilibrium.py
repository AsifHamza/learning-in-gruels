# %% [markdown]
# # Step 5 — How you know it is an equilibrium, and what the equilibrium does
#
# Steps 2, 3 and 4 all called something "the equilibrium" and got on with it. This notebook stops
# and asks the two questions that were taken for granted.
#
# **First: what is actually being checked?** The solver only ever tests a single player changing a
# single action in a single state. That looks too weak to establish anything — surely a player
# could gain by rearranging its plan across several states at once. Section 5.2 is why it is not
# too weak, and it is the most important idea in this folder.
#
# **Second: what does the equilibrium do?** Not the survival numbers it produces — the *policy*.
# Who shoots whom, and under what circumstances. Folder 03 judges its agents on exactly that,
# because an agent can reach the right survival numbers by following a rule that looks nothing
# like the equilibrium.
#
# The solver is step 4's, itself step 3's, copied forward unchanged. Nothing here is random, so
# there is no seed.

# %%
import itertools

ABSTAIN = None

LOW = 0.4
HIGH = 0.8

# %% [markdown]
# ## 5.1 The solver, unchanged
#
# Copied forward. The comments explaining why backward induction is the right shape live in step
# 3; repeating them here would let the two drift apart.

# %%
def accuracy_set(n, lo=LOW, hi=HIGH):
    """Evenly spaced accuracies, smallest first."""
    values = []
    for k in range(n):
        fraction = k / (n - 1)
        values.append(lo + (hi - lo) * fraction)
    return values


def neighbours_on_circle(alive, position):
    """The players `position` can shoot. Just one of them if only two are left."""
    how_many = len(alive)
    index_here = alive.index(position)

    left = alive[(index_here - 1) % how_many]
    right = alive[(index_here + 1) % how_many]

    if left == right:
        return [left]
    return [left, right]


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


def solve_all(accuracies, topology="circle"):
    """Backward induction over every state. Returns values, chosen profiles, and counts."""
    n = len(accuracies)
    known_values = {}
    chosen_profiles = {}
    how_many_equilibria = {}

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


def shoot_strongest_neighbour(alive, shooter, accuracies):
    """Fire at the better shot of your two neighbours."""
    reachable = neighbours_on_circle(alive, shooter)

    best_target = None
    best_accuracy = -1.0
    for candidate in reachable:
        if accuracies[candidate] > best_accuracy:
            best_accuracy = accuracies[candidate]
            best_target = candidate

    return best_target


print("solver copied forward from step 4")

# %% [markdown]
# ## 5.2 Why checking one state at a time is enough
#
# The criterion the solver applies is this:
#
# > **No player can raise its own survival probability by changing its own action in a single
# > state, with everyone else's plan — and its own plan in every other state — left alone.**
#
# That is a *one-shot deviation* test. The worry is obvious: a player might not gain from any
# single change while still gaining from several changes made together. If so, passing this test
# would mean very little.
#
# It does not happen here, for two reasons that are worth checking rather than believing.
#
# **Reason one: a kill strictly shrinks the alive-set, so no state can ever recur.** The state is
# the set of who is alive, and it only ever loses members. So the graph of states is a **directed
# acyclic graph** — there is no path back to a state you have already left, and therefore no way
# to profit from coordinating your own behaviour across two visits to the same situation, because
# there is never a second visit.

# %%
def successors_of(alive, accuracies, choice_of):
    """Every state reachable in one step, given what everyone does."""
    results = []
    for shooter in alive:
        target = choice_of[shooter]
        if target is ABSTAIN:
            continue
        remaining = []
        for player in alive:
            if player != target:
                remaining.append(player)
        if tuple(remaining) not in results:
            results.append(tuple(remaining))
    return results


N_CHECK = 6
accuracies_check = accuracy_set(N_CHECK)
survival_check, profiles_check, counts_check, values_check = solve_all(accuracies_check, "circle")

states_examined = 0
every_successor_smaller = True
worst_case = None

for alive in profiles_check:
    for nxt in successors_of(alive, accuracies_check, profiles_check[alive]):
        states_examined = states_examined + 1
        if len(nxt) != len(alive) - 1:
            every_successor_smaller = False
            worst_case = (alive, nxt)

print(f"N={N_CHECK}: {len(profiles_check)} states, {states_examined} state transitions examined")
print(f"every transition removes exactly one player: {every_successor_smaller}")
if worst_case is not None:
    print(f"  counterexample: {worst_case[0]} -> {worst_case[1]}")
print()
print("So a state is never revisited, and the state graph is a DAG.")
print("There is no second visit to coordinate a plan across.")

# %% [markdown]
# **Reason two: a miss leaves the state unchanged, and that self-loop is already inside the
# value.**
#
# When a shot misses, play continues with exactly the same players alive. Read naively, the value
# of a state then depends on itself, which is an infinite regress.
#
# The recursion sidesteps it by conditioning on **which shot lands next** rather than on how long
# that takes. Each shooter's accuracy acts as a hazard rate; the hazards are normalised so they
# sum to one; and the value is the hazard-weighted average of the successor states' values. The
# waiting time drops out because nothing in this game depends on elapsed time — no discounting,
# no per-round cost.
#
# You can see this in `value_of_profile`: the weights are `hazard[target] / total_hazard`, and
# they sum to 1 across successors. Nothing refers to the current state's own value.

# %%
alive_here = tuple(range(N_CHECK))
profile_here = profiles_check[alive_here]

hazard = {}
for shooter in alive_here:
    target = profile_here[shooter]
    if target is ABSTAIN:
        continue
    hazard[target] = hazard.get(target, 0.0) + accuracies_check[shooter]

total = 0.0
for target in hazard:
    total = total + hazard[target]

print(f"state {alive_here}")
print()
print("  target   hazard aimed at it   share of the next death")
for target in sorted(hazard):
    print(f"    p{target}         {hazard[target]:.4f}              {hazard[target]/total:.4f}")
print()
print(f"  shares sum to {sum(hazard[t]/total for t in hazard):.10f}")
print()
print("Those shares are a probability distribution over SUCCESSOR states.")
print("The value of this state is their weighted average -- it never refers to itself,")
print("so the miss self-loop needed no special handling.")

# %% [markdown]
# ### Putting the two together
#
# Continuations are fixed (they belong to strictly smaller states, already solved) and states
# never recur. So a plan that cannot be improved one state at a time cannot be improved by any
# combination of changes either — there is no interaction between the changes to exploit.
#
# That means passing this test gives something stronger than a Nash equilibrium of the game
# viewed from the opening position. It gives a profile that is optimal **in every state, including
# states that the equilibrium itself makes unlikely to be reached**. The usual name for that is
# **Markov perfect**.
#
# That is the paragraph to be able to say out loud. Everything below is consequences.

# %% [markdown]
# ## 5.3 Running the criterion against shoot-strongest
#
# The most useful thing to point the test at is the heuristic, because steps 1 and 2 treated it
# and the equilibrium as the same object.
#
# A "profitable deviation" is one player, in one state, with one alternative action that would pay
# it more. The unit is the (state, player, alternative) triple: a player with two better actions
# in one state counts twice. Note
# what this count is and is not: it is a **count of opportunities**, not a quality score. More
# players and more states mean more chances for one to exist.

# %%
def deviations_from(policy, n, topology="circle"):
    """Every (state, player, alternative) where the alternative pays that player more than `policy`."""
    accuracies = accuracy_set(n)
    known_values = {}
    for player in range(n):
        values = [0.0] * n
        values[player] = 1.0
        known_values[(player,)] = values

    found = []
    for size in range(2, n + 1):
        for alive in itertools.combinations(range(n), size):
            choice_of = {}
            for shooter in alive:
                choice_of[shooter] = policy(alive, shooter, accuracies)

            values = value_of_profile(accuracies, alive, choice_of, known_values)
            known_values[alive] = values

            for shooter in alive:
                for alternative in options_for(alive, shooter, topology):
                    if alternative == choice_of[shooter]:
                        continue
                    trial = dict(choice_of)
                    trial[shooter] = alternative
                    gained = value_of_profile(accuracies, alive, trial, known_values)[shooter]
                    if gained > values[shooter] + 1e-12:
                        found.append({
                            "state": alive,
                            "player": shooter,
                            "played": choice_of[shooter],
                            "better": alternative,
                            "before": values[shooter],
                            "after": gained,
                        })

    return found


print("   N   profitable deviations from shoot-strongest")
deviation_counts = {}
all_deviations = {}
for n in range(3, 8):
    found = deviations_from(shoot_strongest_neighbour, n)
    deviation_counts[n] = len(found)
    all_deviations[n] = found
    print(f"   {n}        {len(found)}")

print()
equilibrium_at = []
for n in sorted(deviation_counts):
    if deviation_counts[n] == 0:
        equilibrium_at.append(n)
print(f"shoot-strongest is an equilibrium at N = {equilibrium_at}")
print(f"and is not at N = {[n for n in sorted(deviation_counts) if deviation_counts[n] > 0]}")

# %% [markdown]
# ### The worked example
#
# One deviation, printed in full, at N=5 in the opening state. This is the concrete answer to
# "why would anyone shoot the weaker of two neighbours?"

# %%
accuracies_n5 = accuracy_set(5)
opening_n5 = (0, 1, 2, 3, 4)

example = None
for item in all_deviations[5]:
    if item["state"] == opening_n5 and item["player"] == 2:
        example = item

if example is None:
    print("No deviation by p2 in the opening state at N=5, so the worked example below")
    print("does not apply and the numbers would need re-deriving.")
else:
    player = example["player"]
    left, right = neighbours_on_circle(opening_n5, player)
    print(f"N=5, everyone alive, seated in accuracy order {[f'{a:.1f}' for a in accuracies_n5]}")
    print()
    print(f"  p{player} has accuracy {accuracies_n5[player]:.2f}")
    print(f"  its neighbours are p{left} ({accuracies_n5[left]:.2f}) "
          f"and p{right} ({accuracies_n5[right]:.2f})")
    print()
    print(f"  shoot-strongest fires at p{example['played']} "
          f"({accuracies_n5[example['played']]:.2f}): survival {example['before']:.5f}")
    print(f"  firing at p{example['better']} "
          f"({accuracies_n5[example['better']]:.2f}) instead:  survival {example['after']:.5f}")
    print(f"  gain: +{example['after'] - example['before']:.5f}")
    print()
    ratio = example["after"] / example["before"]
    print(f"  which is {ratio:.2f} times the survival, from one changed action")

# %% [markdown]
# **Why that happens.** Firing at your strong neighbour is partly doing your *other* neighbour a
# favour: you remove a threat that was also aimed past you, and you stay sitting next to whoever
# is left. Removing the weak neighbour instead leaves you next to the strong one but shortens the
# circle in your favour. Being surrounded by weak players turns out to be worth more than removing
# a strong one — which is the whole reason the middle of the circle behaves differently from the
# ends.

# %% [markdown]
# ## 5.4 What the equilibrium does — the targeting policy
#
# Now the second question. The quantity is **P(shoot the stronger of your two neighbours)**, and
# it needs two pieces of care.
#
# **Which decisions count.** Three exclusions, and each one would otherwise corrupt the number:
#
# - a player who **abstains** made no targeting decision at all;
# - with **two alive** the left and right neighbour are the *same player*, so there is no stronger
#   one to choose;
# - if both neighbours have the **same accuracy** there is no stronger one either.
#
# **How states are weighted.** Not equally, for two reasons, and both matter because the number has
# to mean the same thing as the one measured on trained agents in folders 03 and 04. There it is the
# share of *decisions*, counted one by one as episodes are played.
#
# - Some states are far more likely than others under the equilibrium's own play, so each state
#   counts in proportion to the probability of **reaching** it.
# - A state that is reached is not left at once. Every step one of its $m$ players is drawn, and
#   the state ends only when a shot hits. If the shooters' accuracies add up to $H$, a step ends the
#   state with probability $H/m$, so the state lasts $m/H$ steps on average, and each player is drawn
#   in $1/m$ of them. So every player makes **$1/H$ decisions per visit**, and a state where shots
#   rarely land produces more decisions than one where they land at once.
#
# Each decision therefore carries weight $\text{reach}/H$. Weighting by reach alone would count a
# state once per visit however long the players spent in it, which is a different quantity from the
# one an agent's decisions measure.

# %%
def state_reach(n, profiles, accuracies):
    """Probability each state is ever visited, under the profiles' own play."""
    reach = {tuple(range(n)): 1.0}

    for size in range(n, 1, -1):
        states_this_size = []
        for alive in reach:
            if len(alive) == size:
                states_this_size.append(alive)

        for alive in states_this_size:
            here = reach[alive]
            if here == 0.0 or alive not in profiles:
                continue

            hazard = {}
            for shooter in alive:
                target = profiles[alive][shooter]
                if target is ABSTAIN:
                    continue
                hazard[target] = hazard.get(target, 0.0) + accuracies[shooter]

            total = 0.0
            for target in hazard:
                total = total + hazard[target]
            if total == 0.0:
                continue

            for target in hazard:
                remaining = []
                for player in alive:
                    if player != target:
                        remaining.append(player)
                nxt = tuple(remaining)
                reach[nxt] = reach.get(nxt, 0.0) + here * (hazard[target] / total)

    return reach


# The weighting is checked against a number already trusted: the probability of reaching
# "only player i is alive" MUST equal i's survival probability.
print("  N   max |reach of 'only p left'  -  survival[p]|")
for n in [3, 5, 7]:
    survival, profiles, _, _ = solve_all(accuracy_set(n), "circle")
    reach = state_reach(n, profiles, accuracy_set(n))
    worst = 0.0
    for player in range(n):
        worst = max(worst, abs(reach.get((player,), 0.0) - survival[player]))
    print(f"  {n}          {worst:.2e}")

print()
print("So the reach numbers are not a separate belief -- they reproduce the survival")
print("vector the earlier steps already established.")

# %% [markdown]
# ### How unequal are the states?

# %%
survival_n7, profiles_n7, _, _ = solve_all(accuracy_set(7), "circle")
reach_n7 = state_reach(7, profiles_n7, accuracy_set(7))

one_death = {}
for alive in reach_n7:
    if len(alive) == 6:
        one_death[alive] = reach_n7[alive]

print("N=7, the states one death in:")
for alive in sorted(one_death, key=lambda s: -one_death[s]):
    missing = [p for p in range(7) if p not in alive][0]
    print(f"  p{missing} died first: probability {one_death[alive]:.4f}")

possible = 7
print()
print(f"  reachable: {len(one_death)} of {possible} possible")
print(f"  spread: {min(one_death.values()):.4f} to {max(one_death.values()):.4f}")
if len(one_death) < possible:
    never = []
    for player in range(7):
        state = tuple(p for p in range(7) if p != player)
        if reach_n7.get(state, 0.0) == 0.0:
            never.append(player)
    print(f"  never dies first under the equilibrium: {never}")
print()
print("Weighting these equally would describe a game nobody plays.")

# %% [markdown]
# ### The targeting policy itself

# %%
def targeting(n, profiles, accuracies, reach, per_decision=True):
    """P(shoot the stronger neighbour), by position and by survivor count.

    With per_decision=True, each decision in a state is weighted by reach / H, where H is the
    sum of the shooters' accuracies in that state: the expected number of decisions each player
    makes per visit. With per_decision=False each state counts once per visit, by reach alone.
    """
    by_position_hit = {}
    by_position_weight = {}
    by_alive_hit = {}
    by_alive_weight = {}

    for alive in profiles:
        here = reach.get(alive, 0.0)
        if here == 0.0:
            continue

        total_hazard = 0.0
        for shooter in alive:
            if profiles[alive][shooter] is not ABSTAIN:
                total_hazard = total_hazard + accuracies[shooter]
        if total_hazard == 0.0:
            continue                                      # nobody shoots, so nobody targets
        if per_decision:
            weight = here / total_hazard                  # decisions per player per visit
        else:
            weight = here                                 # once per visit

        for shooter in alive:
            target = profiles[alive][shooter]
            if target is ABSTAIN:
                continue                                  # made no targeting decision

            reachable = neighbours_on_circle(alive, shooter)
            if len(reachable) < 2:
                continue                                  # left and right are the same player

            first, second = reachable
            if accuracies[first] == accuracies[second]:
                continue                                  # no stronger one to pick

            if accuracies[first] > accuracies[second]:
                stronger = first
            else:
                stronger = second

            hit = 0.0
            if target == stronger:
                hit = 1.0

            by_position_weight[shooter] = by_position_weight.get(shooter, 0.0) + weight
            by_position_hit[shooter] = by_position_hit.get(shooter, 0.0) + weight * hit

            how_many = len(alive)
            by_alive_weight[how_many] = by_alive_weight.get(how_many, 0.0) + weight
            by_alive_hit[how_many] = by_alive_hit.get(how_many, 0.0) + weight * hit

    by_position = {}
    for shooter in by_position_weight:
        by_position[shooter] = by_position_hit[shooter] / by_position_weight[shooter]

    by_alive = {}
    for how_many in by_alive_weight:
        by_alive[how_many] = by_alive_hit[how_many] / by_alive_weight[how_many]

    return by_position, by_alive


print("P(shoot the stronger neighbour), by accuracy rank, weakest first")
print()
targeting_by_n = {}
for n in [3, 5, 7]:
    accuracies = accuracy_set(n)
    survival, profiles, _, _ = solve_all(accuracies, "circle")
    reach = state_reach(n, profiles, accuracies)
    by_position, by_alive = targeting(n, profiles, accuracies, reach)
    targeting_by_n[n] = (by_position, by_alive)

    row = []
    for player in range(n):
        if player in by_position:
            row.append(f"{by_position[player]:.5f}")
        else:
            row.append("  n/a  ")
    print(f"  N={n}   " + "  ".join(row))

# %% [markdown]
# **Why the weighting has to be per decision.** Counting each state once per visit gives different
# numbers, because a state where shots rarely land lasts longer and so holds more decisions. The
# agents in folders 03 and 04 are measured decision by decision, so only the first row below can be
# compared with them.

# %%
accuracies = accuracy_set(5)
survival, profiles, _, _ = solve_all(accuracies, "circle")
reach = state_reach(5, profiles, accuracies)
per_decision, _ = targeting(5, profiles, accuracies, reach, per_decision=True)
per_visit, _ = targeting(5, profiles, accuracies, reach, per_decision=False)
print("N=5, P(shoot the stronger neighbour), weakest first")
print("  per decision  " + "  ".join(f"{per_decision[p]:.5f}" for p in range(5)))
print("  per visit     " + "  ".join(f"{per_visit[p]:.5f}" for p in range(5)))

# %% [markdown]
# ### Reading the shape, by finding it rather than asserting it

# %%
for n in [5, 7]:
    by_position, _ = targeting_by_n[n]
    weakest = by_position[0]
    strongest = by_position[n - 1]
    middles = []
    for player in range(1, n - 1):
        if player in by_position:
            middles.append(by_position[player])

    lower_extreme = min(weakest, strongest)
    highest_middle = max(middles)
    u_shaped = lower_extreme > highest_middle

    print(f"N={n}: weakest {weakest:.4f}, strongest {strongest:.4f}, "
          f"middle {min(middles):.4f}-{max(middles):.4f}")
    print(f"      lower extreme ({lower_extreme:.4f}) above highest middle "
          f"({highest_middle:.4f})? {u_shaped}")

by_position_n3, _ = targeting_by_n[3]
flat_at_n3 = max(by_position_n3.values()) - min(by_position_n3.values())
print()
print(f"N=3: spread across ranks is {flat_at_n3:.4f} -- flat, and that is CORRECT.")
print("     With three alive your two neighbours are the only two opponents, so the")
print("     'stronger neighbour' IS the strongest player and the equilibrium coincides")
print("     with shoot-strongest. Looking for a U-shape at N=3 is the wrong test.")

# %% [markdown]
# ### Targeting depends on how many are left

# %%
not_monotone = []

for n in [5, 7]:
    _, by_alive = targeting_by_n[n]
    print(f"N={n}:")
    for how_many in sorted(by_alive, reverse=True):
        print(f"  {how_many} alive: {by_alive[how_many]:.4f}")

    counts = sorted(by_alive)                      # ascending: 3 alive first
    fewest, most = counts[0], counts[-1]

    # Two separate questions. Does it end higher than it starts, and does it
    # rise at EVERY step? They have different answers, so both get printed.
    ends_higher = by_alive[fewest] > by_alive[most]
    monotone = True
    for index in range(1, len(counts)):
        if by_alive[counts[index]] > by_alive[counts[index - 1]] + 1e-12:
            monotone = False

    print(f"  at {most} alive {by_alive[most]:.4f} -> at {fewest} alive "
          f"{by_alive[fewest]:.4f}")
    print(f"  higher with fewer alive: {ends_higher}")
    print(f"  rises at every step: {monotone}")
    if not monotone:
        not_monotone.append(n)
    print()

endgame = []
for n in [5, 7]:
    _, by_alive = targeting_by_n[n]
    endgame.append(by_alive[min(by_alive)])

print("Every game ends the same way: with three alive the equilibrium fires at the")
print(f"stronger neighbour with probability {min(endgame):.4f}-{max(endgame):.4f},")
print("which is shoot-strongest.")
print()
if len(not_monotone) == 0:
    print("And it rises at every step on the way there.")
else:
    print(f"But it does NOT rise at every step: at N={not_monotone} it dips partway.")
    print("So the defensible statement is that targeting DEPENDS on the survivor")
    print("count and ends at 1.0, not that it climbs steadily toward it.")
print()
print("Either way a policy that ignores how many remain cannot be optimal, which is")
print("why n_alive is part of the observation the agents in folder 03 receive.")

# %% [markdown]
# ## 5.5 Archetti's projection, and what it is worth
#
# There is a published rule for the **complete-graph** version of this game with four or more
# players (Archetti 2012): shoot the type immediately weaker than yourself, unless you are the
# weakest, in which case shoot the strongest. On a circle you can only reach two players, so the
# rule has to be projected onto what is actually reachable:
#
# > **Shoot the strongest of your neighbours who is still weaker than you. If both are stronger
# > than you, shoot the stronger one.**
#
# The second clause is Archetti's rule recovered locally — if everyone reachable outguns you, you
# are the weak one and you fire at the strongest.
#
# It is worth measuring because it is the **only** reference in this project with the
# equilibrium's U-shape, and above N=8 no exact answer is on record to compare an agent against.
# Under ranked seating the circle stays sorted as players die, so this rule always shoots the
# left-hand neighbour; folder 04 measures it under the name `next_weaker`.
# It is also worth being honest about, which is what this section is for.

# %%
def archetti_target(alive, shooter, accuracies):
    """Strongest reachable player still weaker than you; else the stronger one."""
    reachable = neighbours_on_circle(alive, shooter)

    weaker = []
    for candidate in reachable:
        if accuracies[candidate] < accuracies[shooter]:
            weaker.append(candidate)

    if len(weaker) > 0:
        best = weaker[0]
        for candidate in weaker:
            if accuracies[candidate] > accuracies[best]:
                best = candidate
        return best

    best = reachable[0]
    for candidate in reachable:
        if accuracies[candidate] > accuracies[best]:
            best = candidate
    return best


def profiles_of_policy(n, accuracies, policy):
    """What a fixed policy does in every state, in the same shape solve_all returns."""
    profiles = {}
    for size in range(2, n + 1):
        for alive in itertools.combinations(range(n), size):
            choice_of = {}
            for shooter in alive:
                choice_of[shooter] = policy(alive, shooter, accuracies)
            profiles[alive] = choice_of
    return profiles


print("Each policy is weighted by ITS OWN reach, because different play visits")
print("different states. Mixing one policy's weights with another's gives a wrong")
print("answer that looks reasonable.")
print()
print("   N   Archetti mean gap   worst single rank   shoot-strongest mean gap   ratio")

archetti_gaps = {}
for n in range(3, 8):
    accuracies = accuracy_set(n)
    _, equilibrium_profiles, _, _ = solve_all(accuracies, "circle")
    equilibrium_reach = state_reach(n, equilibrium_profiles, accuracies)
    equilibrium_targeting, _ = targeting(n, equilibrium_profiles, accuracies, equilibrium_reach)

    archetti_profiles = profiles_of_policy(n, accuracies, archetti_target)
    archetti_reach = state_reach(n, archetti_profiles, accuracies)
    archetti_targeting, _ = targeting(n, archetti_profiles, accuracies, archetti_reach)

    heuristic_profiles = profiles_of_policy(n, accuracies, shoot_strongest_neighbour)
    heuristic_reach = state_reach(n, heuristic_profiles, accuracies)
    heuristic_targeting, _ = targeting(n, heuristic_profiles, accuracies, heuristic_reach)

    shared = []
    for player in range(n):
        if (player in equilibrium_targeting and player in archetti_targeting
                and player in heuristic_targeting):
            shared.append(player)

    archetti_errors = []
    heuristic_errors = []
    for player in shared:
        archetti_errors.append(abs(archetti_targeting[player] - equilibrium_targeting[player]))
        heuristic_errors.append(abs(heuristic_targeting[player] - equilibrium_targeting[player]))

    archetti_mean = sum(archetti_errors) / len(archetti_errors)
    heuristic_mean = sum(heuristic_errors) / len(heuristic_errors)
    archetti_gaps[n] = {
        "mean": archetti_mean,
        "worst": max(archetti_errors),
        "heuristic_mean": heuristic_mean,
        "archetti_targeting": archetti_targeting,
        "equilibrium_targeting": equilibrium_targeting,
    }

    if heuristic_mean <= 1e-12:
        # shoot-strongest IS the equilibrium here, so "times further away" is
        # meaningless -- it is not further away at all.
        ratio = "ss exact"
    elif archetti_mean > 1e-12:
        ratio = f"{heuristic_mean / archetti_mean:.1f}x"
    else:
        ratio = "arch exact"
    print(f"   {n}       {archetti_mean:.4f}              {max(archetti_errors):.4f}"
          f"             {heuristic_mean:.4f}                {ratio}")

# %% [markdown]
# ### Three honest limits
#
# Each one derived from the table above rather than asserted.

# %%
# 1. Does it converge?
above_n3 = [n for n in sorted(archetti_gaps) if n > 3]
first_gap = archetti_gaps[above_n3[0]]["mean"]
last_gap = archetti_gaps[above_n3[-1]]["mean"]
print(f"1. Does the approximation improve with N?")
print(f"   gap at N={above_n3[0]}: {first_gap:.4f}   gap at N={above_n3[-1]}: {last_gap:.4f}")
shrinking = last_gap < first_gap * 0.5
print(f"   materially shrinking? {shrinking}")
print(f"   -> it is closer than shoot-strongest, but it does NOT converge.")
print()

# 2. Which ranks does it get wrong?
print("2. Where is the error? N=5, middle ranks:")
detail = archetti_gaps[5]
print("   rank   equilibrium   Archetti   ")
for player in [1, 2, 3]:
    print(f"    {player}      {detail['equilibrium_targeting'][player]:.4f}       "
          f"{detail['archetti_targeting'][player]:.4f}")
under = 0
for player in [1, 2, 3]:
    if detail["archetti_targeting"][player] < detail["equilibrium_targeting"][player]:
        under = under + 1
print(f"   ranks where Archetti sits BELOW the equilibrium: {under} of 3")
shares = []
for player in [1, 2, 3]:
    shares.append(detail["archetti_targeting"][player] / detail["equilibrium_targeting"][player])
print(f"   -> right about the direction, {min(shares):.2f} to {max(shares):.2f} of the magnitude.")
print("      It UNDERSHOOTS the middle, which is exactly where the interesting behaviour is.")
print()

# 3. N=3
print("3. At N=3, where Archetti's four-player rule is applied outside its range:")
detail3 = archetti_gaps[3]
print("   rank   equilibrium   Archetti")
for player in sorted(detail3["equilibrium_targeting"]):
    print(f"    {player}      {detail3['equilibrium_targeting'][player]:.4f}       "
          f"{detail3['archetti_targeting'][player]:.4f}")
print(f"   worst single-rank error: {detail3['worst']:.4f}")
print(f"   shoot-strongest's error at N=3: {detail3['heuristic_mean']:.4f}")
print("   -> QUALITATIVELY WRONG: the projection misses the middle player by the largest")
print("      error possible, while shoot-strongest is exactly right. Archetti's own")
print("      three-player rule is different and agrees with the equilibrium.")

# %% [markdown]
# So Archetti's value is narrow and real: **where no exact answer is on record there is no
# targeting reference at all**, and this is the only candidate with the right shape. It should be
# quoted with its gap attached, never as ground truth.

# %% [markdown]
# ## 5.6 One assumption worth checking
#
# Everything above reads a policy off "the" equilibrium. If a state had several equilibria, that
# policy would be partly an artefact of which one the solver picked. Step 4 established that no
# state in the solvable range has more than one — but it is checked here too, because this is the
# section whose conclusions depend on it.

# %%
total_states = 0
states_with_a_choice = 0
for n in range(3, 9):
    _, _, counts, _ = solve_all(accuracy_set(n), "circle")
    for alive in counts:
        total_states = total_states + 1
        if counts[alive] > 1:
            states_with_a_choice = states_with_a_choice + 1

print(f"states across N=3 to N=8: {total_states}")
print(f"states with more than one equilibrium: {states_with_a_choice}")
print()
if states_with_a_choice == 0:
    print("The equilibrium is unique everywhere it exists, so the targeting policy above")
    print("is not a selection artefact -- there was never a choice to make.")
else:
    print("Some states offered a choice, so the targeting policy above is partly a")
    print("consequence of the tie-break rule and should be read with that in mind.")

# %% [markdown]
# ## 5.7 The targeting profiles, written out in full
#
# Section 5.4 rounds these to five decimals. This prints each one at twelve, one line per rank, in
# a fixed format, so any quotation of them elsewhere can be checked against what this notebook
# computed.

# %%
for n in sorted(targeting_by_n):
    by_position, _ = targeting_by_n[n]
    for rank in sorted(by_position):
        print(f"targeting N={n} rank {rank} P(shoot stronger) {by_position[rank]:.12f}")

# %% [markdown]
# ## What to take from step 5
#
# - The criterion is a **one-shot deviation** test: no player gains by changing one action in one
#   state, everything else fixed.
# - It is **sufficient, not just necessary**, for two reasons that were checked here rather than
#   assumed. A kill strictly shrinks the alive-set, so the state graph is a **DAG** and no state
#   recurs — there is no second visit across which to coordinate a better plan. And a miss leaves
#   the state unchanged, a self-loop the recursion already absorbs by conditioning on **which shot
#   lands next** and averaging over successors with hazard weights that sum to 1.
# - Passing therefore means **Markov perfect** — optimal in every state, including the ones the
#   equilibrium itself rarely reaches — not merely Nash from the opening position.
# - **Shoot-strongest is an equilibrium at N=3 and nowhere above it**, with 3 / 18 / 67 / 201
#   profitable deviations at N=4 / 5 / 6 / 7. Those counts measure opportunities, not badness.
# - The worked case: at N=5 the player on 0.60 between 0.50 and 0.70 gains **+0.04752** survival
#   by shooting the *weaker* neighbour — because being surrounded by weak players is worth more
#   than removing a strong one.
# - The equilibrium's **targeting policy is U-shaped**: extremes fire at their stronger neighbour
#   almost always, the middle mostly fires at its weaker one. Neither `shoot_strongest` nor a coin
#   flip produces that shape. At N=3 the shape is flat and that is correct, not a failure.
# - Targeting **depends on the survivor count**, reaching 1.0000 with three alive at every N —
#   though not by rising at every step, since at N=7 it dips partway down. Either way a policy
#   ignoring the survivor count cannot be optimal.
# - All of it is weighted **per decision**, the way agents are measured: by how likely each state is
#   to be reached, verified against the survival vector the earlier steps already established, and
#   by how many decisions are made there per visit.
# - **Archetti's projection** is 1.5 to 3.3 times closer than shoot-strongest and is the only
#   reference with the right shape, but it does not converge and it undershoots the middle. Applied
#   at N=3, outside the range Archetti gives it for, it misses the middle player by a full 1.0.
#
# That completes deliverable 02. Folder `03-agent-architecture/` trains agents that have to
# rediscover these numbers without being told them, and measures this same targeting quantity to
# find out what they actually learned.
