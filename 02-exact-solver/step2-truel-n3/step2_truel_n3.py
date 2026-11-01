# %% [markdown]
# # Step 2 — Three players, and the number 8/27
#
# Step 1 gave us one idea: **throw away the steps where nothing happens, and weight what can
# happen by hazard.** This notebook applies it to three players and lands exactly on the
# three numbers the whole project is built around.
#
# We do it twice. First by hand, on paper, for N=3 specifically — so you can see there is no
# magic. Then with code that works for any accuracies, which we check against six published
# results from a 1997 paper.

# %%
# Nothing random here, so no seed. Nothing imported either.
ACCURACIES = {"A": 0.8, "B": 0.6, "C": 0.4}

print("A is the best shot:", ACCURACIES["A"])
print("B is in the middle:", ACCURACIES["B"])
print("C is the worst shot:", ACCURACIES["C"])

# %% [markdown]
# ## 2.1 Who shoots whom
#
# At N=3 the circle is a triangle, so everybody can reach everybody. The strategy is **shoot
# your strongest opponent**, because the biggest threat to you is the best shot still alive.
#
# So with A=0.8, B=0.6, C=0.4:
#
# | Player | Strongest opponent | Shoots |
# |---|---|---|
# | A (0.8) | B (0.6) | B |
# | B (0.6) | A (0.8) | A |
# | C (0.4) | A (0.8) | A |
#
# **Two guns are pointed at A and only one at B.** Nobody at all is pointed at C. Being the
# best shot has made A the most dangerous person to leave alive, and being the worst has made
# C invisible. That is the whole mechanism, and it is already visible before any arithmetic.

# %% [markdown]
# ## 2.2 Who dies first
#
# Each step, one of the three is picked at random, so each fires with probability 1/3. The
# chance per step that a particular player lands a hit is $\tfrac13 \times$ their accuracy.
#
# - **B dies** if A fires and hits: $\tfrac13 \times 0.8$
# - **A dies** if B hits *or* C hits: $\tfrac13 \times 0.6 + \tfrac13 \times 0.4 = \tfrac13 \times 1.0$
# - **C dies**: never, nobody is shooting at C
#
# Using step 1's rule, drop the nothing-happens steps and take the ratio. The $\tfrac13$ appears
# in every term so it cancels:
#
# $$P(\text{B first}) = \frac{0.8}{0.8 + 1.0} = \frac{0.8}{1.8} = \frac49,
# \qquad P(\text{A first}) = \frac{1.0}{1.8} = \frac59$$

# %%
hazard_on_B = ACCURACIES["A"]                          # only A shoots B
hazard_on_A = ACCURACIES["B"] + ACCURACIES["C"]        # both B and C shoot A
total_hazard = hazard_on_A + hazard_on_B

chance_B_dies_first = hazard_on_B / total_hazard
chance_A_dies_first = hazard_on_A / total_hazard

print("hazard aimed at A:", hazard_on_A)
print("hazard aimed at B:", hazard_on_B)
print("hazard aimed at C:", 0.0)
print()
print(f"B dies first: {chance_B_dies_first:.6f}  = 4/9 ? {abs(chance_B_dies_first - 4/9) < 1e-12}")
print(f"A dies first: {chance_A_dies_first:.6f}  = 5/9 ? {abs(chance_A_dies_first - 5/9) < 1e-12}")

# %% [markdown]
# ## 2.3 Then it is a duel
#
# After the first death, two players are left, and step 1 already told us the answer:
# $W(p,q) = p/(p+q)$.
#
# **If B died first** (probability 4/9), A and C duel:
#
# $$W(0.8, 0.4) = \tfrac{0.8}{1.2} = \tfrac23 \quad\text{for A},
# \qquad W(0.4, 0.8) = \tfrac13 \quad\text{for C}$$
#
# **If A died first** (probability 5/9), B and C duel:
#
# $$W(0.6, 0.4) = \tfrac{0.6}{1.0} = \tfrac35 \quad\text{for B},
# \qquad W(0.4, 0.6) = \tfrac25 \quad\text{for C}$$

# %%
def duel(p, q):
    return p / (p + q)


A, B, C = ACCURACIES["A"], ACCURACIES["B"], ACCURACIES["C"]

print("if B dies first, A beats C with probability:", duel(A, C))
print("if B dies first, C beats A with probability:", duel(C, A))
print("if A dies first, B beats C with probability:", duel(B, C))
print("if A dies first, C beats B with probability:", duel(C, B))

# %% [markdown]
# ## 2.4 Multiply out
#
# $$P_A = \tfrac49 \cdot \tfrac23 = \tfrac{8}{27}$$
#
# $$P_B = \tfrac59 \cdot \tfrac35 = \tfrac{3}{9} = \tfrac{9}{27}$$
#
# $$P_C = \tfrac49 \cdot \tfrac13 + \tfrac59 \cdot \tfrac25 = \tfrac{4}{27} + \tfrac{2}{9} = \tfrac{10}{27}$$
#
# C appears twice because C survives the first death either way — nobody is shooting at C, so
# C always makes it to the duel, and only then has to fight.

# %%
P_A = chance_B_dies_first * duel(A, C)
P_B = chance_A_dies_first * duel(B, C)
P_C = chance_B_dies_first * duel(C, A) + chance_A_dies_first * duel(C, B)

print("player   survival        as a fraction of 27")
print(f"  A      {P_A:.10f}    {P_A * 27:.6f} / 27")
print(f"  B      {P_B:.10f}    {P_B * 27:.6f} / 27")
print(f"  C      {P_C:.10f}    {P_C * 27:.6f} / 27")
print()
print("adds to:", P_A + P_B + P_C)
print()
print("P_A == 8/27 :", abs(P_A - 8 / 27) < 1e-15)
print("P_B == 9/27 :", abs(P_B - 9 / 27) < 1e-15)
print("P_C == 10/27:", abs(P_C - 10 / 27) < 1e-15)

# %% [markdown]
# **8, 9, 10 out of 27.** The worst shot is the most likely to survive, and the three
# probabilities are consecutive integers over 27, which is the kind of coincidence that makes
# you trust the arithmetic.
#
# Compare with folder 01, which needed 100,000 random episodes to get 0.3705 and still had a
# wobble of 0.0015. This took no episodes at all and is exact.

# %% [markdown]
# ## 2.5 Now write it as code that works for any accuracies
#
# The hand version is convincing but useless for anything else. The general version is a
# **recursion**: to find the value of a state, look at every way somebody could die, weight
# each by its hazard, and ask for the value of the smaller state that follows.
#
# The base case is one player left: they survive with probability 1.
#
# In symbols, writing $S$ for the set of players still alive and $h_t$ for the total hazard
# aimed at player $t$:
#
# $$V_i(S) = \sum_{t \in S} \frac{h_t}{\sum_{u} h_u} \; V_i(S \setminus \{t\})$$
#
# That is the entire solver. Everything else below is bookkeeping.
#
# Two things are quietly doing a lot of work:
#
# **Missed shots do not appear anywhere.** They are the "nothing happens" steps, and step 1
# showed they cancel.
#
# **Neither does the 1/3 for whose turn it is.** Every hazard has the same factor, so it
# divides out of the ratio.

# %%
def solve(accuracies, policy, topology="complete"):
    """Exact survival probability for every player, under a fixed policy.

    accuracies: list of accuracies, one per position.
    policy: function(alive, shooter, accuracies) -> target position, or None to abstain.
    """
    n = len(accuracies)
    cache = {}

    def targets_of(alive, shooter):
        """Who this player is allowed to shoot."""
        if topology == "complete":
            allowed = []
            for other in alive:
                if other != shooter:
                    allowed.append(other)
            return allowed

        # circle: only the two neighbours in the current order
        how_many = len(alive)
        index_here = alive.index(shooter)
        left = alive[(index_here - 1) % how_many]
        right = alive[(index_here + 1) % how_many]
        if left == right:
            return [left]
        return [left, right]

    def value(alive):
        """Survival probability of every player, given this set is still alive."""
        if alive in cache:
            return cache[alive]

        # Base case: one player left, they have won.
        if len(alive) == 1:
            result = [0.0] * n
            result[alive[0]] = 1.0
            cache[alive] = result
            return result

        # Add up the hazard pointed at each player.
        hazard = {}
        for shooter in alive:
            target = policy(alive, shooter, accuracies)
            if target is None:
                continue
            if target not in hazard:
                hazard[target] = 0.0
            hazard[target] = hazard[target] + accuracies[shooter]

        total_hazard = 0.0
        for target in hazard:
            total_hazard = total_hazard + hazard[target]

        # If nobody is shooting, nobody ever dies. The game is stuck and the
        # prize is split between everyone still alive.
        if total_hazard == 0.0:
            share = 1.0 / len(alive)
            result = [0.0] * n
            for player in alive:
                result[player] = share
            cache[alive] = result
            return result

        # Otherwise: average the smaller states, weighted by hazard.
        result = [0.0] * n
        for target in hazard:
            weight = hazard[target] / total_hazard

            remaining = []
            for player in alive:
                if player != target:
                    remaining.append(player)
            remaining = tuple(remaining)

            smaller = value(remaining)
            for player in range(n):
                result[player] = result[player] + weight * smaller[player]

        cache[alive] = result
        return result

    everyone = tuple(range(n))
    survival = value(everyone)
    return survival, cache


def shoot_strongest(alive, shooter, accuracies):
    """Fire at whoever has the highest accuracy among the players you can reach."""
    best_target = None
    best_accuracy = -1.0

    for other in alive:
        if other == shooter:
            continue
        if accuracies[other] > best_accuracy:
            best_accuracy = accuracies[other]
            best_target = other

    return best_target


# %% [markdown]
# ## 2.6 Does the code agree with the hand derivation?

# %%
# Position 0 is the strongest, to match A, B, C above.
accuracies = [0.8, 0.6, 0.4]
survival, cache = solve(accuracies, shoot_strongest)

print("            by hand         by code        difference")
for index, name in enumerate(["A", "B", "C"]):
    by_hand = [P_A, P_B, P_C][index]
    print(f"  {name} ({accuracies[index]})   {by_hand:.12f}   {survival[index]:.12f}   "
          f"{abs(by_hand - survival[index]):.2e}")

print()
print("states the solver had to evaluate:", len(cache))
print("adds to:", sum(survival))

# %% [markdown]
# ## 2.7 The real test: six published rows
#
# Agreeing with my own hand derivation proves very little — both could be wrong in the same
# way. The proper test is against numbers somebody else published.
#
# Kilgour & Brams, *The Truel*, Mathematics Magazine 70(5), 1997, page 317, tabulates six
# accuracy triples for exactly this game. **Three of the six reverse the survival ordering**,
# which is what makes them a good test: a solver that reproduces all six cannot have been
# accidentally fitted to one memorised answer.
#
# Their numbers are printed to three decimal places, so agreement to about 0.0005 is the best
# that can be expected.

# %%
PUBLISHED = [
    # a,    b,    c,     P_A,    P_B,    P_C
    (0.9, 0.5, 0.10, 0.540, 0.333, 0.127),
    (0.9, 0.5, 0.30, 0.397, 0.294, 0.309),
    (0.8, 0.7, 0.20, 0.376, 0.412, 0.212),
    (0.6, 0.5, 0.25, 0.314, 0.370, 0.316),
    (0.8, 0.5, 0.40, 0.314, 0.294, 0.392),
    (0.8, 0.6, 0.40, 0.296, 0.333, 0.370),   # this is our case
]

print("  a     b     c   |  published P_A P_B P_C  |  computed              | worst error")
print("-" * 90)

worst_error_overall = 0.0

for row in PUBLISHED:
    a, b, c, published_A, published_B, published_C = row

    computed, _ = solve([a, b, c], shoot_strongest)

    errors = [
        abs(computed[0] - published_A),
        abs(computed[1] - published_B),
        abs(computed[2] - published_C),
    ]
    worst_error = max(errors)
    if worst_error > worst_error_overall:
        worst_error_overall = worst_error

    print(f" {a:.2f}  {b:.2f}  {c:.2f}  |  "
          f"{published_A:.3f} {published_B:.3f} {published_C:.3f}     |  "
          f"{computed[0]:.5f} {computed[1]:.5f} {computed[2]:.5f}  |  {worst_error:.5f}")

print()
print(f"worst disagreement across all six rows: {worst_error_overall:.6f}")
if worst_error_overall < 0.001:
    print("=> every row matches to the precision they were printed at.")
else:
    print("=> a row does NOT match. The solver is wrong.")

# %% [markdown]
# ## 2.8 Why those three reversed rows matter
#
# Look at the orderings:
#
# | Accuracies | Ordering | Weakest survives best? |
# |---|---|---|
# | 0.9, 0.5, 0.1 | $P_A > P_B > P_C$ | **no** — strongest wins |
# | 0.9, 0.5, 0.3 | $P_A > P_C > P_B$ | no |
# | 0.8, 0.7, 0.2 | $P_B > P_A > P_C$ | no |
# | 0.6, 0.5, 0.25 | $P_B > P_C > P_A$ | partly |
# | 0.8, 0.5, 0.4 | $P_C > P_A > P_B$ | yes |
# | 0.8, 0.6, 0.4 | $P_C > P_B > P_A$ | **yes** — fully reversed |
#
# **Survival-of-the-weakest is not a law.** With 0.9, 0.5, 0.1 the best shot simply wins.
# The effect needs the accuracies to be reasonably close together: if the weakest is *far*
# too weak, being ignored is not enough to save them.
#
# Our case, 0.8 / 0.6 / 0.4, sits at the far end where the reversal is complete. That is not
# an accident — it is the case chosen for the project because it is where the interesting
# thing happens.

# %% [markdown]
# ## 2.9 Is shoot-strongest actually the right strategy?
#
# So far we *assumed* everyone shoots their strongest opponent. That assumption is doing a
# lot of work, and it needs justifying — otherwise 8/27 is just "what happens under one
# particular guess".
#
# The test is whether it is an **equilibrium**: can any single player do better for
# themselves by switching, given the others do not change?
#
# In folder 01 we tested this by simulation with three candidate deviations. Here we can test
# **every** possible strategy, exactly. With 3 players each choosing between 2 targets and
# abstaining, that is only $3^3 = 27$ combinations.

# %%
ABSTAIN = None


def value_of_profile(accuracies, alive, choice_of, continuation):
    """Survival probabilities when each living player fires at their chosen target."""
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

    if total_hazard == 0.0:
        # Nobody shooting: stuck, prize split.
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
        smaller = continuation[tuple(remaining)]
        for player in range(n):
            result[player] = result[player] + weight * smaller[player]
    return result


def find_equilibria(accuracies, alive, continuation):
    """Every combination of choices where nobody can improve by switching alone."""
    n = len(accuracies)
    players = list(alive)

    # What each player may do: shoot any opponent, or abstain.
    options = {}
    for shooter in players:
        choices = []
        for other in players:
            if other != shooter:
                choices.append(other)
        choices.append(ABSTAIN)
        options[shooter] = choices

    # Build every combination. Written as nested loops over an index list so it is
    # readable rather than clever.
    all_combinations = [{}]
    for shooter in players:
        grown = []
        for partial in all_combinations:
            for choice in options[shooter]:
                extended = dict(partial)
                extended[shooter] = choice
                grown.append(extended)
        all_combinations = grown

    equilibria = []

    for combination in all_combinations:
        values = value_of_profile(accuracies, alive, combination, continuation)
        is_stable = True

        for shooter in players:
            for alternative in options[shooter]:
                if alternative == combination[shooter]:
                    continue
                trial = dict(combination)
                trial[shooter] = alternative
                trial_values = value_of_profile(accuracies, alive, trial, continuation)
                if trial_values[shooter] > values[shooter] + 1e-12:
                    is_stable = False
                    break
            if not is_stable:
                break

        if is_stable:
            equilibria.append((combination, values))

    return equilibria


# %%
accuracies = [0.8, 0.6, 0.4]

# Continuations for the two-player states, which we already know exactly.
continuation = {}
for pair in [(0, 1), (0, 2), (1, 2)]:
    first, second = pair
    values = [0.0, 0.0, 0.0]
    values[first] = duel(accuracies[first], accuracies[second])
    values[second] = duel(accuracies[second], accuracies[first])
    continuation[pair] = values
for single in [(0,), (1,), (2,)]:
    values = [0.0, 0.0, 0.0]
    values[single[0]] = 1.0
    continuation[single] = values

equilibria = find_equilibria(accuracies, (0, 1, 2), continuation)

print(f"combinations checked: 27")
print(f"equilibria found: {len(equilibria)}")
print()

names = {0: "A(0.8)", 1: "B(0.6)", 2: "C(0.4)", ABSTAIN: "abstain"}

for combination, values in equilibria:
    description = []
    for shooter in [0, 1, 2]:
        description.append(f"{names[shooter]} -> {names[combination[shooter]]}")
    print("  " + " | ".join(description))
    print(f"     survival: A={values[0]:.5f}  B={values[1]:.5f}  C={values[2]:.5f}")

# %% [markdown]
# ## 2.10 What that tells us
#
# There is exactly **one** equilibrium, it is "everybody shoots their strongest opponent",
# and its survival probabilities are 8/27, 9/27, 10/27.
#
# So the target the whole project uses is not one arbitrary policy's outcome. It is *the*
# outcome, the only one nobody has a reason to deviate from.
#
# **And nobody abstains in it.** That is worth dwelling on, because the project's original
# proposal expected the opposite. The old idea — "the weakest player should fire in the air to
# avoid becoming a target" — comes from a *different* version of this game, one where players
# take turns in a fixed order. When the order is random, as it is here, abstaining is simply
# a wasted turn. The equilibrium abstention probability is exactly zero, which is what folder
# 01's focal-agent table showed by simulation and what this cell shows by exhaustion.

# %%
# Check the claim directly: is all-abstain an equilibrium?
all_abstain = {0: ABSTAIN, 1: ABSTAIN, 2: ABSTAIN}
abstain_values = value_of_profile(accuracies, (0, 1, 2), all_abstain, continuation)

print("if everybody abstains, each gets:", abstain_values)
print()

for shooter in [0, 1, 2]:
    best_gain = 0.0
    best_choice = None
    for other in [0, 1, 2]:
        if other == shooter:
            continue
        trial = dict(all_abstain)
        trial[shooter] = other
        trial_values = value_of_profile(accuracies, (0, 1, 2), trial, continuation)
        gain = trial_values[shooter] - abstain_values[shooter]
        if gain > best_gain:
            best_gain = gain
            best_choice = other

    if best_choice is None:
        print(f"  {names[shooter]}: cannot improve")
    else:
        print(f"  {names[shooter]}: switching to shoot {names[best_choice]} "
              f"gains {best_gain:+.5f}")

print()
print("Everyone gains by breaking ranks, so all-abstain is not an equilibrium.")

# %% [markdown]
# ## 2.11 Is the middle player always 1/3?
#
# Folder 01 noticed something odd: the middle player survived almost exactly 1/3 of the time
# under four completely different policies. Here we can find out whether that is a real fact
# or a coincidence of the numbers 0.8 / 0.6 / 0.4.

# %%
print("   a     b     c   |   P_A      P_B      P_C    | is P_B exactly 1/3?")
print("-" * 72)

trials = [
    [0.8, 0.6, 0.4],
    [0.9, 0.5, 0.1],
    [0.8, 0.7, 0.2],
    [0.7, 0.5, 0.3],
    [0.95, 0.55, 0.15],
    [0.6, 0.55, 0.5],
]

for trial in trials:
    computed, _ = solve(trial, shoot_strongest)
    is_third = abs(computed[1] - 1 / 3) < 1e-9
    print(f" {trial[0]:.2f}  {trial[1]:.2f}  {trial[2]:.2f}  |  "
          f"{computed[0]:.5f}  {computed[1]:.5f}  {computed[2]:.5f}  |  {is_third}")

# %% [markdown]
# So it is **not** a general law — it depends on the accuracies. Look at what makes it happen.
#
# The middle player B survives only if A dies first *and* B then wins the duel against C:
#
# $$P_B = \frac{b + c}{(b+c) + a} \cdot \frac{b}{b + c}$$
#
# The $(b+c)$ cancels, leaving
#
# $$P_B = \frac{b}{a + b + c}$$
#
# and that equals $1/3$ exactly when $b$ is one third of the total — which is to say when the
# three accuracies are **evenly spaced**, since then the middle one is the mean. The project
# uses evenly spaced accuracies by construction, so 1/3 shows up every time. Change the
# spacing and it goes away.

# %%
print("checking P_B = b / (a + b + c):")
for trial in trials:
    a, b, c = trial
    computed, _ = solve(trial, shoot_strongest)
    formula = b / (a + b + c)
    print(f"  {trial}: solver {computed[1]:.10f}, formula {formula:.10f}, "
          f"match {abs(computed[1] - formula) < 1e-12}")

# %% [markdown]
# ## What to take from step 2
#
# - The exact triple is **8/27, 9/27, 10/27**, derived two ways that agree to 12 decimals.
# - The mechanism is visible before any arithmetic: **two guns point at the best shot, none at
#   the worst.**
# - The general solver is one formula — average the smaller states, weighted by hazard — and
#   the misses and the whose-turn factor both cancel out of it.
# - It reproduces **all six** published rows to the precision they were printed at.
# - Three of those six do **not** show survival-of-the-weakest, so it is a consequence of the
#   accuracies being close together, not a law of the game.
# - Shoot-strongest is the **unique** equilibrium at N=3, and **nobody abstains in it**.
# - The middle player's 1/3 is $b/(a+b+c)$, which is 1/3 only for evenly spaced accuracies.
#
# Next: `step3_circle_n5.py`, where the graph finally starts to matter.
