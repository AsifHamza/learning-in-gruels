# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Step 11 — Can an actor that sees four numbers play the equilibrium?
#
# The actor sees four numbers: its own accuracy, its two neighbours' accuracies, and the share
# of the field still alive. The equilibrium is computed from the whole set of survivors. Two
# different states can therefore look identical to a player. If the equilibrium asks for
# different actions in two states that look the same, no actor of this kind can play the
# equilibrium exactly, however long it trains.
#
# This step answers three questions at five and seven players, under ranked seating:
#
# 1. How often do states that look identical to a player call for different equilibrium actions?
# 2. What is the most any four-number policy could agree with the equilibrium, decision by decision?
# 3. How much do the trained actors of steps 6 and 9 actually agree with it, decision by decision?
#
# It trains nothing. It solves the game exactly, and it loads the saved actors of steps 6 and 9,
# checking each against the hash in its record. Nothing here is random, so there is no seed to
# repeat a run with.

# %%
import hashlib
import itertools
import json

import numpy as np
import torch
import torch.nn as nn

from gruel_env import GruelEnv, MISS, SHOOT_LEFT, SHOOT_RIGHT, N_ACTIONS

SEED = 0
PLAYER_COUNTS = [5, 7]
torch.set_num_threads(1)
print("torch", torch.__version__)

# %% [markdown]
# ## The solver, copied from folder 02 step 5
#
# The functions below are folder 02's solver, copied rather than imported. One change: the set
# of survivors is called `survivors` here, where folder 02 calls it `alive`. The solver is
# handed the survivors because computing the equilibrium needs them; the actor, further down,
# is handed only its four numbers.

# %%
ABSTAIN = None
LOW = 0.4
HIGH = 0.8


def accuracy_set(n, lo=LOW, hi=HIGH):
    """Evenly spaced accuracies, smallest first."""
    values = []
    for k in range(n):
        fraction = k / (n - 1)
        values.append(lo + (hi - lo) * fraction)
    return values


def neighbours_on_circle(survivors, position):
    """The players `position` can shoot. Just one of them if only two are left."""
    how_many = len(survivors)
    index_here = survivors.index(position)
    left = survivors[(index_here - 1) % how_many]
    right = survivors[(index_here + 1) % how_many]
    if left == right:
        return [left]
    return [left, right]


def value_of_profile(accuracies, survivors, choice_of, known_values):
    """Survival probability for everyone, when each living player shoots their choice."""
    n = len(accuracies)
    hazard = {}
    for shooter in survivors:
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
        share = 1.0 / len(survivors)
        result = [0.0] * n
        for player in survivors:
            result[player] = share
        return result

    result = [0.0] * n
    for target in hazard:
        weight = hazard[target] / total_hazard
        remaining = []
        for player in survivors:
            if player != target:
                remaining.append(player)
        smaller = known_values[tuple(remaining)]
        for player in range(n):
            result[player] = result[player] + weight * smaller[player]
    return result


def options_for(survivors, shooter):
    """Shoot a reachable opponent, or abstain."""
    choices = list(neighbours_on_circle(survivors, shooter))
    choices.append(ABSTAIN)
    return choices


def every_combination(survivors):
    combinations = [{}]
    for shooter in survivors:
        grown = []
        for partial in combinations:
            for choice in options_for(survivors, shooter):
                extended = dict(partial)
                extended[shooter] = choice
                grown.append(extended)
        combinations = grown
    return combinations


def find_equilibria(accuracies, survivors, known_values):
    equilibria = []
    for combination in every_combination(survivors):
        values = value_of_profile(accuracies, survivors, combination, known_values)
        is_stable = True
        for shooter in survivors:
            for alternative in options_for(survivors, shooter):
                if alternative == combination[shooter]:
                    continue
                trial = dict(combination)
                trial[shooter] = alternative
                trial_values = value_of_profile(accuracies, survivors, trial, known_values)
                if trial_values[shooter] > values[shooter] + 1e-12:
                    is_stable = False
                    break
            if not is_stable:
                break
        if is_stable:
            equilibria.append((combination, values))
    return equilibria


def solve_all(accuracies):
    """Backward induction over every state. Returns values, chosen profiles and counts."""
    n = len(accuracies)
    known_values = {}
    chosen_profiles = {}
    how_many_equilibria = {}
    for player in range(n):
        values = [0.0] * n
        values[player] = 1.0
        known_values[(player,)] = values
    for size in range(2, n + 1):
        for survivors in itertools.combinations(range(n), size):
            equilibria = find_equilibria(accuracies, survivors, known_values)
            if len(equilibria) == 0:
                raise RuntimeError(f"no equilibrium at state {survivors}")
            how_many_equilibria[survivors] = len(equilibria)
            combination, values = equilibria[0]
            known_values[survivors] = values
            chosen_profiles[survivors] = combination
    return known_values, chosen_profiles, how_many_equilibria


def state_reach(n, profiles, accuracies):
    """Probability each state is ever visited, under the profiles' own play."""
    reach = {tuple(range(n)): 1.0}
    for size in range(n, 1, -1):
        states_this_size = []
        for survivors in reach:
            if len(survivors) == size:
                states_this_size.append(survivors)
        for survivors in states_this_size:
            here = reach[survivors]
            hazard = {}
            for shooter in survivors:
                target = profiles[survivors][shooter]
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
                for player in survivors:
                    if player != target:
                        remaining.append(player)
                nxt = tuple(remaining)
                reach[nxt] = reach.get(nxt, 0.0) + here * (hazard[target] / total)
    return reach


# %% [markdown]
# ## Solve both games, and check they are the games folder 02 solved
#
# Every state must have exactly one pure equilibrium, as folder 02 found, so taking the first
# is not a choice. The weakest player's exact survival is printed for comparison with folder 02.

# %%
SOLVED = {}
for n in PLAYER_COUNTS:
    accuracies = accuracy_set(n)
    values, profiles, counts = solve_all(accuracies)
    unique = True
    for survivors in counts:
        if counts[survivors] != 1:
            unique = False
    reach = state_reach(n, profiles, accuracies)
    SOLVED[n] = {"accuracies": accuracies, "values": values, "profiles": profiles, "reach": reach}
    weakest = values[tuple(range(n))][0]
    print("N=%d  states %d  every state has exactly one pure equilibrium: %s  "
          "weakest player's exact survival %.5f" % (n, len(counts), unique, weakest))

# %% [markdown]
# ## What a player sees, written the way the environment writes it
#
# For each state and each player alive, the observation is built from the solver's state, and
# the equilibrium's action is turned into the environment's action: shoot left, shoot right or
# miss. "Left" is the previous survivor in seat order, as in `GruelEnv.neighbours`.
#
# States with two players alive are left out: both shooting actions then point at the same
# opponent, so there is no choice between neighbours to get wrong.
#
# Each decision is weighted by how often it is made under equilibrium play: the probability of
# reaching the state, times $1/H$, where $H$ is the total hazard in that state. That is the
# weighting folder 02 step 5 uses for targeting.

# %%
def observation_in(n, accuracies, survivors, player):
    index_here = survivors.index(player)
    left = survivors[(index_here - 1) % len(survivors)]
    right = survivors[(index_here + 1) % len(survivors)]
    numbers = [accuracies[player], accuracies[left], accuracies[right], len(survivors) / n]
    return np.array(numbers, dtype=np.float32), left, right


def equilibrium_action(target, left, right):
    if target is ABSTAIN:
        return MISS
    if target == left:
        return SHOOT_LEFT
    if target == right:
        return SHOOT_RIGHT
    raise ValueError("target is not a neighbour")


def total_hazard(accuracies, survivors, profile):
    total = 0.0
    for shooter in survivors:
        if profile[shooter] is not ABSTAIN:
            total = total + accuracies[shooter]
    return total


DECISIONS = {}
for n in PLAYER_COUNTS:
    solved = SOLVED[n]
    rows = []
    for survivors in solved["profiles"]:
        if len(survivors) < 3:
            continue
        reach = solved["reach"].get(survivors, 0.0)
        hazard = total_hazard(solved["accuracies"], survivors, solved["profiles"][survivors])
        for player in survivors:
            observation, left, right = observation_in(n, solved["accuracies"], survivors, player)
            action = equilibrium_action(solved["profiles"][survivors][player], left, right)
            weight = 0.0
            if hazard > 0.0:
                weight = reach / hazard
            rows.append({"survivors": survivors, "player": player, "observation": observation,
                         "action": action, "weight": weight})
    DECISIONS[n] = rows
    print("N=%d  state-player pairs with three or more alive: %d" % (n, len(rows)))

# %% [markdown]
# ### Check: the observations match the environment's
#
# The environment is reset under ranked seating and players are removed from its seat order by
# hand. For every state its `observe` must return exactly the observation built above.

# %%
matches = 0
checked = 0
for n in PLAYER_COUNTS:
    game = GruelEnv(n=n, reward="terminal", seating="ranked", seed=SEED)
    for row in DECISIONS[n]:
        game.reset()
        game.order = list(row["survivors"])
        from_environment = game.observe(game.agent_of_position[row["player"]])
        checked = checked + 1
        if np.array_equal(from_environment, row["observation"]):
            matches = matches + 1
print("observations checked against the environment: %d, identical: %d" % (checked, matches))

# %% [markdown]
# ## 1. Do identical observations call for different actions?
#
# Decisions are grouped by their observation. A group **conflicts** if the equilibrium asks for
# more than one action within it. The **ceiling** is the most any policy that sees only the
# observation could agree with the equilibrium: in each group, playing the action that carries
# the most weight.

# %%
ACTION_NAMES = {SHOOT_LEFT: "left", SHOOT_RIGHT: "right", MISS: "miss"}


def group_by_observation(rows):
    groups = {}
    for row in rows:
        key = tuple(round(float(value), 6) for value in row["observation"])
        if key not in groups:
            groups[key] = []
        groups[key].append(row)
    return groups


CEILING = {}
GROUPS = {}
for n in PLAYER_COUNTS:
    groups = group_by_observation(DECISIONS[n])
    GROUPS[n] = groups
    total_weight = 0.0
    conflicting_weight = 0.0
    best_weight = 0.0
    conflicting_groups = 0
    for key in groups:
        weight_by_action = {}
        for row in groups[key]:
            weight_by_action[row["action"]] = weight_by_action.get(row["action"], 0.0) + row["weight"]
            total_weight = total_weight + row["weight"]
        best_weight = best_weight + max(weight_by_action.values())
        if len(weight_by_action) > 1:
            conflicting_groups = conflicting_groups + 1
            for action in weight_by_action:
                conflicting_weight = conflicting_weight + weight_by_action[action]
    CEILING[n] = best_weight / total_weight
    print("N=%d  distinct observations %d  conflicting %d  "
          "share of decisions in a conflicting observation %.4f  ceiling %.4f"
          % (n, len(groups), conflicting_groups, conflicting_weight / total_weight, CEILING[n]))

# %% [markdown]
# ### The conflicts, heaviest first
#
# Each line is one observation that calls for different actions in different states. The
# states are the survivors by seat, weakest at 0; the action is the one the equilibrium asks of
# the observing player there.

# %%
for n in PLAYER_COUNTS:
    print("N=%d" % n)
    listed = []
    for key in GROUPS[n]:
        actions = set(row["action"] for row in GROUPS[n][key])
        if len(actions) > 1:
            weight = sum(row["weight"] for row in GROUPS[n][key])
            listed.append((weight, key))
    listed.sort(reverse=True)
    if len(listed) == 0:
        print("  none")
    for weight, key in listed:
        own, left, right, share = key
        print("  own %.4f  left %.4f  right %.4f  alive %d   weight %.4f"
              % (own, left, right, round(share * n), weight))
        for row in GROUPS[n][key]:
            print("      survivors %-24s player %d   equilibrium: %-5s  weight %.4f"
                  % (str(row["survivors"]), row["player"], ACTION_NAMES[row["action"]], row["weight"]))
    print()

# %% [markdown]
# ## 2. How closely do the trained actors agree, decision by decision?
#
# The saved actors are loaded from steps 6 and 9, and each one's weights are checked against the
# hash in its record. For every state and player, the actor's probability of the equilibrium's
# action is read straight from its network. The **agreement** is that probability averaged over
# decisions with the equilibrium weights above; it can be compared directly with the ceiling.

# %%
OBSERVATION_SIZE = 4
HIDDEN_SIZE = 64


class SharedActorCritic(nn.Module):
    def __init__(self, n_players):
        super().__init__()
        self.n_players = n_players
        self.actor_first = nn.Linear(OBSERVATION_SIZE, HIDDEN_SIZE)
        self.actor_second = nn.Linear(HIDDEN_SIZE, N_ACTIONS)
        critic_input_size = 2 * n_players + n_players
        self.critic_first = nn.Linear(critic_input_size, HIDDEN_SIZE)
        self.critic_second = nn.Linear(HIDDEN_SIZE, 1)
        self.activation = nn.Tanh()

    def action_logits(self, observation):
        hidden = self.activation(self.actor_first(observation))
        return self.actor_second(hidden)


def weights_sha256(model):
    """The same hash the training steps write into each record."""
    digest = hashlib.sha256()
    weights = model.state_dict()
    for name in sorted(weights):
        tensor = weights[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(list(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


SAVED = {
    5: [("step6-n5-gate", seed) for seed in range(5)],
    7: [("step9-player-count", seed) for seed in range(3)],
}

# Step 9's five-player seeds 0 to 2 are the same configuration as step 6's seeds 0 to 2.
SAME_AS_STEP6 = [("step9-player-count", seed) for seed in range(3)]


def load_actor(n, folder, seed):
    run = "n%d_ranked_terminal_centralised_u2000_seed%d" % (n, seed)
    path = "../%s/models/%s" % (folder, run)
    with open(path + ".json") as handle:
        record = json.load(handle)
    model = SharedActorCritic(n_players=n)
    model.load_state_dict(torch.load(path + ".pt"))
    model.eval()
    return model, weights_sha256(model) == record["weights_sha256"]


def action_probabilities(model, observation):
    with torch.no_grad():
        logits = model.action_logits(torch.from_numpy(observation))
    return torch.softmax(logits, dim=-1).numpy()


AGREEMENT = {}
for n in PLAYER_COUNTS:
    rows = DECISIONS[n]
    total_weight = sum(row["weight"] for row in rows)
    print("N=%d   ceiling %.4f" % (n, CEILING[n]))
    print("    model                       hash   agreement   most likely action is the equilibrium's")
    AGREEMENT[n] = []
    for folder, seed in SAVED[n]:
        model, hash_ok = load_actor(n, folder, seed)
        agreement = 0.0
        modal = 0.0
        for row in rows:
            probabilities = action_probabilities(model, row["observation"])
            agreement = agreement + row["weight"] * float(probabilities[row["action"]])
            if int(np.argmax(probabilities)) == row["action"]:
                modal = modal + row["weight"]
        agreement = agreement / total_weight
        modal = modal / total_weight
        AGREEMENT[n].append(agreement)
        print("    %-20s seed %d   %-5s  %.4f      %.4f" % (folder, seed, hash_ok, agreement, modal))
    print("    mean agreement %.4f, lowest %.4f, highest %.4f"
          % (sum(AGREEMENT[n]) / len(AGREEMENT[n]), min(AGREEMENT[n]), max(AGREEMENT[n])))
    print()

# %% [markdown]
# ### Check: step 9's five-player actors are step 6's
#
# Step 9 trained the five-player configuration again with seeds 0 to 2. Its saved weights should
# be identical to step 6's, which is why they are not counted twice above.

# %%
for folder, seed in SAME_AS_STEP6:
    theirs, ok_theirs = load_actor(5, folder, seed)
    ours, ok_ours = load_actor(5, "step6-n5-gate", seed)
    print("seed %d: step 9 and step 6 weights identical: %s" % (seed, weights_sha256(theirs) == weights_sha256(ours)))

# %% [markdown]
# ### Where the agreement is lost, by players alive
#
# The ceiling and the agreement again, but for each number of players alive separately, with the
# share of all decisions made at that size. At seven players this shows whether the actors fall
# short where the observations conflict, or elsewhere.

# %%
for n in PLAYER_COUNTS:
    print("N=%d" % n)
    print("    alive   share of decisions   ceiling   agreement, mean over seeds")
    rows = DECISIONS[n]
    total_weight = sum(row["weight"] for row in rows)
    models = []
    for folder, seed in SAVED[n]:
        model, hash_ok = load_actor(n, folder, seed)
        models.append(model)
    for alive in range(n, 2, -1):
        here = [row for row in rows if len(row["survivors"]) == alive]
        weight_here = sum(row["weight"] for row in here)
        groups = group_by_observation(here)
        best = 0.0
        for key in groups:
            weight_by_action = {}
            for row in groups[key]:
                weight_by_action[row["action"]] = weight_by_action.get(row["action"], 0.0) + row["weight"]
            best = best + max(weight_by_action.values())
        agreements = []
        for model in models:
            agreement = 0.0
            for row in here:
                agreement = agreement + row["weight"] * float(action_probabilities(model, row["observation"])[row["action"]])
            agreements.append(agreement / weight_here)
        print("    %d       %.4f               %.4f    %.4f"
              % (alive, weight_here / total_weight, best / weight_here, sum(agreements) / len(agreements)))
    print()

# %% [markdown]
# ## What this step establishes, derived from the measurements

# %%
for n in PLAYER_COUNTS:
    mean_agreement = sum(AGREEMENT[n]) / len(AGREEMENT[n])
    print("N=%d: a four-number policy can agree with the equilibrium on at most %.4f of decisions;"
          " the trained actors agree on %.4f on average, %.4f of the ceiling."
          % (n, CEILING[n], mean_agreement, mean_agreement / CEILING[n]))
