# %% [markdown]
# # Step 2 — Can one player do better by switching?
#
# Folder 02 defines an equilibrium as a set of strategies where **no single player can do better
# by changing what it does on its own**. That is a test, not a description, and it does not need
# the exact answer to run. It only needs two measurements of the same seat:
#
# 1. its expected return when everyone plays the trained agent, and
# 2. its expected return when everyone else still plays the agent but that one seat switches to
#    something else.
#
# If the second is clearly higher, the agent's policy is **not** an equilibrium: a player has a
# reason to abandon it. That conclusion is firm. If no switch helps, the policy has survived the
# switches that were tried, which is evidence rather than proof, because only four alternatives
# are tried and an equilibrium must resist every possible one.
#
# This is the one check in the folder that asks the same question as folder 02, at player counts
# folder 02 cannot reach.

# %%
import time

import numpy as np
import torch
import torch.nn as nn

from gruel_env import GruelEnv, N_ACTIONS, MISS, SHOOT_LEFT, SHOOT_RIGHT

torch.set_num_threads(1)
print("torch", torch.__version__)

# %% [markdown]
# ## Settings
#
# The agent is trained exactly as in step 1: same code, same budget, same seeds. Five players is
# included because folder 02 has the exact answer there, which lets the test be checked before it
# is trusted.
#
# Under the `terminal` reward a seat's expected return is its chance of winning, plus its share
# of the prize in the rare episode that stalls, so "does better" here means exactly what it means
# in folder 02.

# %%
PLAYER_COUNTS = [5, 10, 15]
REWARD = "terminal"
SEATING = "ranked"

OBSERVATION_SIZE = 4
HIDDEN_SIZE = 64

UPDATES = 2000
EPISODES_PER_UPDATE = 96
SEEDS = [0, 1, 2]

LEARNING_RATE = 0.003
CLIP = 0.2
EPOCHS = 4
MINIBATCH_SIZE = 256
VALUE_COEFFICIENT = 0.5
ENTROPY_COEFFICIENT = 0.01

BASELINE_EPISODES = 10000
SWITCH_EPISODES = 5000
STANDARD_ERRORS = 3

print("player counts:", PLAYER_COUNTS, " seeds:", SEEDS)
print("episodes: %d for the baseline, %d for each switch" % (BASELINE_EPISODES, SWITCH_EPISODES))
print("a switch counts as profitable only if it gains more than %d standard errors" % STANDARD_ERRORS)

# %% [markdown]
# ## Why three standard errors, not two
#
# Every seat is tried against four switches, so at 15 players one seed runs 60 separate tests. At a
# two-standard-error threshold, chance alone would flag about one of those as a gain. At three, the
# expected number of false alarms across the whole notebook is well under one. The cost is that a
# small real gain can be missed, which is the safer direction for a test whose positive result is
# the firm one.

# %%
tests_per_seed = 0
for n in PLAYER_COUNTS:
    tests_per_seed = tests_per_seed + 4 * n
total_tests = tests_per_seed * len(SEEDS)
# One-sided chance of exceeding 2 and 3 standard errors under a normal approximation.
chance_two = 0.02275
chance_three = 0.00135
print("switch tests in this notebook:", total_tests)
print("expected false alarms at 2 standard errors: %.1f" % (total_tests * chance_two))
print("expected false alarms at 3 standard errors: %.2f" % (total_tests * chance_three))

# %% [markdown]
# ## The four alternatives a seat can switch to
#
# The same four simple rules folder 01 built. `coin_flip` shoots a random neighbour, so it needs its
# own random stream.

# %%
switch_rng = np.random.default_rng(0)


def shoot_strongest(game, actor):
    left, right = game.neighbours(actor)
    if game.accuracies[left] >= game.accuracies[right]:
        return SHOOT_LEFT
    return SHOOT_RIGHT


def shoot_weakest(game, actor):
    left, right = game.neighbours(actor)
    if game.accuracies[left] < game.accuracies[right]:
        return SHOOT_LEFT
    return SHOOT_RIGHT


def coin_flip(game, actor):
    if switch_rng.random() < 0.5:
        return SHOOT_LEFT
    return SHOOT_RIGHT


def always_miss(game, actor):
    return MISS


SWITCHES = [("shoot_strongest", shoot_strongest), ("shoot_weakest", shoot_weakest),
            ("coin_flip", coin_flip), ("always_miss", always_miss)]

# %% [markdown]
# ## The measurement
#
# `mean_returns` plays episodes in which each seat's move comes from whichever policy
# `policy_for_seat` names for it, and returns every seat's average total reward. With `ranked`
# seating, seat 0 is always the weakest player and the last seat the strongest.

# %%
def mean_returns(game, policy_for_seat, episodes):
    totals = []
    for position in range(game.n):
        totals.append(0.0)

    for _ in range(episodes):
        game.reset()
        for acting_agent in game.agent_iter():
            observation, reward, termination, truncation, info = game.last()
            if termination or truncation:
                game.step(None)
                continue
            actor = game.position_of_agent[acting_agent]
            policy = policy_for_seat(actor)
            game.step(policy(game, actor))

        for position in range(game.n):
            name = game.agent_of_position[position]
            totals[position] = totals[position] + game.episode_returns[name]

    averages = []
    for position in range(game.n):
        averages.append(totals[position] / episodes)
    return averages


def standard_error(probability, episodes):
    probability = min(max(probability, 0.0), 1.0)
    return (probability * (1 - probability) / episodes) ** 0.5


def judge(staying, switching):
    """Gain from switching, its standard error, and whether it clears the threshold."""
    gain = switching - staying
    error = (standard_error(staying, BASELINE_EPISODES) ** 2
             + standard_error(switching, SWITCH_EPISODES) ** 2) ** 0.5
    return gain, error, gain > STANDARD_ERRORS * error


# %% [markdown]
# ## Checking the test before trusting it
#
# Two cases at five players where folder 02 already knows the answer.
#
# **First, a policy that is not an equilibrium.** Folder 02 shows that when everyone shoots their
# stronger neighbour, some players can do better by deviating. A working test should find at least
# one profitable switch here.

# %%
def everyone_plays(policy):
    def choose(actor):
        return policy
    return choose


def one_seat_switches(seat, switch_policy, everyone_else):
    def choose(actor):
        if actor == seat:
            return switch_policy
        return everyone_else
    return choose


calibration_game = GruelEnv(n=5, reward=REWARD, seating=SEATING, seed=300)
staying_returns = mean_returns(calibration_game, everyone_plays(shoot_strongest), BASELINE_EPISODES)

found_in_calibration = 0
print("everyone plays shoot_strongest at N=5; each seat tries each switch\n")
print("  seat  accuracy  staying   best switch        switching   gain     verdict")
for seat in range(5):
    best_name = None
    best = None
    for name, policy in SWITCHES:
        switching = mean_returns(calibration_game,
                                 one_seat_switches(seat, policy, shoot_strongest),
                                 SWITCH_EPISODES)[seat]
        gain, error, profitable = judge(staying_returns[seat], switching)
        if best is None or gain > best[0]:
            best = (gain, error, profitable, switching)
            best_name = name
        if profitable:
            found_in_calibration = found_in_calibration + 1
    verdict = "PROFITABLE" if best[2] else "no clear gain"
    print("  %d     %.1f       %.4f   %-17s  %.4f      %+.4f   %s"
          % (seat, 0.4 + 0.1 * seat, staying_returns[seat], best_name, best[3], best[0], verdict))

print()
print("profitable switches found:", found_in_calibration)
if found_in_calibration > 0:
    print("-> the test detects a policy that is not an equilibrium, as folder 02 says it should")
else:
    print("-> the test found nothing where folder 02 says there is something: do not trust it below")

# %% [markdown]
# The second calibration case, a policy that *is* close to an equilibrium, needs a trained agent,
# so it comes after training: at five players the agent matches the exact equilibrium (deliverable
# 03, step 6), so a working test should find few or no profitable switches there.

# %% [markdown]
# ## The agent and its training, copied forward
#
# Every definition below is step 1's, unchanged. Rule 1 says a step copies the previous step
# forward rather than importing it, so this folder carries its own copy and can be handed
# over on its own.

# %%
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

    def value(self, state, agent_one_hot):
        combined = torch.cat([state, agent_one_hot], dim=-1)
        hidden = self.activation(self.critic_first(combined))
        return self.critic_second(hidden).squeeze(-1)


def global_state(game):
    numbers = []
    for position in range(game.n):
        numbers.append(game.accuracies[position])
        if position in game.order:
            numbers.append(1.0)
        else:
            numbers.append(0.0)
    return np.array(numbers, dtype=np.float32)


def one_hot(position, n_players):
    vector = np.zeros(n_players, dtype=np.float32)
    vector[position] = 1.0
    return vector


def collect_batch(model, game, episodes):
    observations = {p: [] for p in range(game.n)}
    actions = {p: [] for p in range(game.n)}
    old_log_probabilities = {p: [] for p in range(game.n)}
    values = {p: [] for p in range(game.n)}
    returns = {p: [] for p in range(game.n)}
    states = {p: [] for p in range(game.n)}
    agent_flags = {p: [] for p in range(game.n)}

    for _ in range(episodes):
        game.reset()
        episode_observations = {p: [] for p in range(game.n)}
        episode_actions = {p: [] for p in range(game.n)}
        episode_log_probabilities = {p: [] for p in range(game.n)}
        episode_values = {p: [] for p in range(game.n)}
        episode_states = {p: [] for p in range(game.n)}
        episode_flags = {p: [] for p in range(game.n)}
        # What each player had already banked when it made each decision. Needed because
        # the return for a decision is whatever arrives AFTER it, and under `shaped`
        # rewards arrive throughout the episode rather than only at the end.
        episode_banked_before = {p: [] for p in range(game.n)}

        for acting_agent in game.agent_iter():
            observation, reward, termination, truncation, info = game.last()

            # A finished agent is handed to us one last time, so PettingZoo can retire
            # it. Stepping with None is how we say 'nothing to do here'.
            if termination or truncation:
                game.step(None)
                continue

            actor = game.position_of_agent[acting_agent]
            banked_so_far = float(game.episode_returns[acting_agent])
            observation_tensor = torch.from_numpy(observation)
            state_array = global_state(game)
            agent_array = one_hot(actor, game.n)

            with torch.no_grad():
                logits = model.action_logits(observation_tensor)
                value = model.value(torch.from_numpy(state_array),
                                    torch.from_numpy(agent_array))

            distribution = torch.distributions.Categorical(logits=logits)
            action = distribution.sample()

            episode_observations[actor].append(observation)
            episode_actions[actor].append(int(action))
            episode_log_probabilities[actor].append(float(distribution.log_prob(action)))
            episode_values[actor].append(float(value))
            episode_states[actor].append(state_array)
            episode_flags[actor].append(agent_array)
            episode_banked_before[actor].append(banked_so_far)

            game.step(int(action))

        for position in range(game.n):
            # The return for a decision is the reward-to-go: everything this player ended
            # the episode with, minus what it had already banked when it decided. With a
            # discount of 1 that is exactly the undiscounted sum of what came after.
            #
            # Reading it from the environment is the whole point of this step. Assuming
            # `1.0 for the winner, 0.0 for everyone else` is correct under `terminal` and
            # silently WRONG under `shaped`, where it would discard every kill bonus,
            # every elimination penalty and every survival tick -- and both arms would then
            # train on identical numbers and look identical for a reason that is a bug.
            name = game.agent_of_position[position]
            final_total = float(game.episode_returns[name])
            for index in range(len(episode_actions[position])):
                own_return = final_total - episode_banked_before[position][index]
                observations[position].append(episode_observations[position][index])
                actions[position].append(episode_actions[position][index])
                old_log_probabilities[position].append(
                    episode_log_probabilities[position][index])
                values[position].append(episode_values[position][index])
                states[position].append(episode_states[position][index])
                agent_flags[position].append(episode_flags[position][index])
                returns[position].append(own_return)

    flat_observations = []
    flat_actions = []
    flat_old_log_probabilities = []
    flat_values = []
    flat_returns = []
    flat_states = []
    flat_flags = []
    for position in range(game.n):
        flat_observations.extend(observations[position])
        flat_actions.extend(actions[position])
        flat_old_log_probabilities.extend(old_log_probabilities[position])
        flat_values.extend(values[position])
        flat_returns.extend(returns[position])
        flat_states.extend(states[position])
        flat_flags.extend(agent_flags[position])

    batch = {
        "observations": torch.tensor(np.array(flat_observations), dtype=torch.float32),
        "actions": torch.tensor(flat_actions, dtype=torch.long),
        "old_log_probabilities": torch.tensor(flat_old_log_probabilities,
                                              dtype=torch.float32),
        "values": torch.tensor(flat_values, dtype=torch.float32),
        "returns": torch.tensor(flat_returns, dtype=torch.float32),
        "states": torch.tensor(np.array(flat_states), dtype=torch.float32),
        "agent_flags": torch.tensor(np.array(flat_flags), dtype=torch.float32),
    }
    return batch


def ppo_update(model, optimiser, batch, update_rng):
    advantages = batch["returns"] - batch["values"]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    count = len(batch["actions"])
    order = np.arange(count)

    for epoch in range(EPOCHS):
        update_rng.shuffle(order)
        for start in range(0, count, MINIBATCH_SIZE):
            chunk = order[start:start + MINIBATCH_SIZE]
            index = torch.from_numpy(chunk)

            logits = model.action_logits(batch["observations"][index])
            distribution = torch.distributions.Categorical(logits=logits)
            new_log_probabilities = distribution.log_prob(batch["actions"][index])

            ratio = torch.exp(new_log_probabilities
                              - batch["old_log_probabilities"][index])
            minibatch_advantages = advantages[index]
            unclipped = ratio * minibatch_advantages
            clipped = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * minibatch_advantages
            policy_loss = -torch.min(unclipped, clipped).mean()

            entropy = distribution.entropy().mean()

            predicted_value = model.value(batch["states"][index],
                                          batch["agent_flags"][index])
            value_loss = ((predicted_value - batch["returns"][index]) ** 2).mean()

            total_loss = (policy_loss
                          + VALUE_COEFFICIENT * value_loss
                          - ENTROPY_COEFFICIENT * entropy)

            optimiser.zero_grad()
            total_loss.backward()
            optimiser.step()


def make_agent_policy(model):
    def agent_policy(game, actor):
        name = game.agent_of_position[actor]
        observation = game.observe(name)
        with torch.no_grad():
            logits = model.action_logits(torch.from_numpy(observation))
        distribution = torch.distributions.Categorical(logits=logits)
        return int(distribution.sample())
    return agent_policy


def accuracies_of(n):
    game = GruelEnv(n=n, reward=REWARD, seating=SEATING, seed=0)
    game.reset(seed=0)
    return [round(float(a), 4) for a in game.accuracies]


def train_one(n, seed):
    torch.manual_seed(2000 + seed)
    game = GruelEnv(n=n, reward=REWARD, seating=SEATING, seed=1000 + seed)
    model = SharedActorCritic(n_players=n)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    update_rng = np.random.default_rng(seed)

    for update in range(UPDATES):
        batch = collect_batch(model, game, EPISODES_PER_UPDATE)
        ppo_update(model, optimiser, batch, update_rng)
    return model

# %% [markdown]
# ## Saving the trained models
#
# Every model this step trains for its results is written to `models/` as two files: the weights
# (`<run>.pt`, a PyTorch `state_dict`) and a record of the model (`<run>.json`): its layers and
# parameter counts, what its actor and critic see, every training setting and seed, the game it was
# trained in, the software versions, and hashes of the weights and of the code that produced them.
# Models trained only to check reproducibility are not saved.
#
# **Two hashes, for two jobs.** `weights_sha256` is computed from the weight values themselves, so
# two runs that produce the same model give the same hash, in this step or any other. `file_sha256`
# is the hash of the `.pt` file, which PyTorch does not write byte-for-byte the same way twice, so
# it checks only that the file has not changed since it was saved.
#
# Saving reads no random stream, so it cannot change any number this notebook prints.

# %%
import datetime
import hashlib
import json
import os
import platform
import subprocess
import time

import gymnasium
import numpy as np
import pettingzoo

MODELS_FOLDER = "models"
NOTEBOOK = "step2_switch_test.ipynb"
SAVED_MODELS = []


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        digest.update(handle.read())
    return digest.hexdigest()


def weights_sha256(model):
    """A hash of the weight values, in name order, so the same model always gives the same hash."""
    digest = hashlib.sha256()
    weights = model.state_dict()
    for name in sorted(weights):
        tensor = weights[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(list(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def git_commit():
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True)
        return result.stdout.strip()
    except Exception:
        return None


def layer_table(model):
    rows = []
    weights = model.state_dict()
    for name in weights:
        rows.append({"name": name, "shape": list(weights[name].shape),
                     "parameters": int(weights[name].numel())})
    return rows


def save_model(model, run_key, settings):
    """Write the weights and the model's record. `settings` holds everything specific to the run."""
    os.makedirs(MODELS_FOLDER, exist_ok=True)
    weights_path = os.path.join(MODELS_FOLDER, run_key + ".pt")
    torch.save(model.state_dict(), weights_path)

    layers = layer_table(model)
    actor_parameters = 0
    critic_parameters = 0
    for row in layers:
        if "critic" in row["name"]:
            critic_parameters = critic_parameters + row["parameters"]
        else:
            actor_parameters = actor_parameters + row["parameters"]

    card = {
        "run": run_key,
        "step": os.path.basename(os.getcwd()),
        "deliverable": os.path.basename(os.path.dirname(os.getcwd())),
        "notebook": NOTEBOOK,
        "model_class": type(model).__name__,
        "layers": layers,
        "activation": "tanh",
        "actor_parameters": actor_parameters,
        "critic_parameters": critic_parameters,
        "total_parameters": actor_parameters + critic_parameters,
        "actions": ["shoot_left", "shoot_right", "miss"],
        "files": {"weights": run_key + ".pt", "record": run_key + ".json"},
        "weights_sha256": weights_sha256(model),
        "file_sha256": file_sha256(weights_path),
        "code": {
            "notebook_source_sha256": file_sha256(NOTEBOOK.replace(".ipynb", ".py")),
            "gruel_env_sha256": file_sha256("gruel_env.py"),
            "git_commit_when_trained": git_commit(),
            "note": "the code hashes pin the exact source; the commit may predate uncommitted edits",
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "gymnasium": gymnasium.__version__,
            "pettingzoo": pettingzoo.__version__,
            "torch_threads": torch.get_num_threads(),
        },
        "machine": {"system": platform.system(), "release": platform.release(),
                    "processor": platform.machine()},
        "saved_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    for key in settings:
        card[key] = settings[key]

    with open(os.path.join(MODELS_FOLDER, run_key + ".json"), "w") as handle:
        json.dump(card, handle, indent=1)
    SAVED_MODELS.append(run_key)
    return weights_path


def run_key(n, seating, reward, critic, updates, seed):
    """One naming scheme in every step, so the same configuration has the same name everywhere."""
    return "n%d_%s_%s_%s_u%d_seed%d" % (n, seating, reward, critic, updates, seed)


def mappo_settings(n, reward, seating, seed, updates, episodes_per_update, critic, seconds,
                   seconds_cover, learning_rate=None, extra=None):
    """The record for a model trained by this folder's PPO loop."""
    if learning_rate is None:
        learning_rate = LEARNING_RATE
    settings = {
        "algorithm": "PPO with a shared actor and a " + critic + " critic",
        "observation": "own accuracy, left neighbour's accuracy, right neighbour's accuracy, share of players alive",
        "critic_input": critic_description(critic, n),
        "game": {"n_players": n, "reward": reward, "seating": seating,
                 "accuracies_ascending": accuracies_ascending(n), "stagnation_rounds": 10},
        "training": {
            "updates": updates,
            "episodes_per_update": episodes_per_update,
            "training_episodes": updates * episodes_per_update,
            "optimiser": "Adam",
            "learning_rate": learning_rate,
            "clip": CLIP,
            "epochs": EPOCHS,
            "minibatch_size": MINIBATCH_SIZE,
            "value_coefficient": VALUE_COEFFICIENT,
            "entropy_coefficient": ENTROPY_COEFFICIENT,
            "discount": 1.0,
            "advantage": "return minus the critic's value, normalised per batch",
        },
        "seeds": {"seed": seed, "torch": 2000 + seed, "training_game": 1000 + seed,
                  "minibatch_shuffle": seed},
        "seconds": round(seconds, 1),
        "seconds_cover": seconds_cover,
    }
    if extra is not None:
        for key in extra:
            settings[key] = extra[key]
    return settings


def accuracies_ascending(n):
    """Read from the environment itself; it uses its own generator, so no shared stream moves."""
    probe = GruelEnv(n=n, reward="terminal", seating="ranked", seed=0)
    probe.reset(seed=0)
    return [round(float(a), 6) for a in probe.accuracies]


def critic_description(critic, n):
    if critic == "centralised":
        return ("every seat's accuracy and alive flag (2N numbers) plus a one-hot of the player "
                "being scored (N numbers): %d inputs" % (3 * n))
    if critic == "local":
        return "the same four numbers as the actor: 4 inputs"
    return critic

# %% [markdown]
# ## Training, as in step 1
#
# One model per player count per seed, from scratch, with the same seeds as step 1. Nothing in
# `train_one` knows which count it is training at.

# %%
TRAINED = {}
started_all = time.time()
for n in PLAYER_COUNTS:
    TRAINED[n] = {}
    for seed in SEEDS:
        started = time.time()
        TRAINED[n][seed] = train_one(n, seed)
        save_model(TRAINED[n][seed], run_key(n, SEATING, REWARD, "centralised", UPDATES, seed),
                   mappo_settings(n, REWARD, SEATING, seed, UPDATES, EPISODES_PER_UPDATE,
                                  "centralised", time.time() - started, "training only"))
        print("  N=%-3d seed %d trained in %4.1f min" % (n, seed, (time.time() - started) / 60.0),
              flush=True)
print("  total %.1f min" % ((time.time() - started_all) / 60.0))

# %% [markdown]
# ## The switch test on the trained agent
#
# For every player count, every seed and every seat: measure the seat's return when everyone plays
# the agent, then when that seat alone switches to each of the four alternatives.
#
# Two random streams matter here, as in every step that measures a trained policy: the
# environment's, and torch's global stream, which the agent samples its actions from. Both are reset
# before every measurement, so each number can be reproduced on its own.

# %%
def agent_for_everyone(agent_policy):
    def choose(actor):
        return agent_policy
    return choose


RESULTS = {}
started_all = time.time()
for n in PLAYER_COUNTS:
    RESULTS[n] = {}
    for seed in SEEDS:
        agent_policy = make_agent_policy(TRAINED[n][seed])
        game = GruelEnv(n=n, reward=REWARD, seating=SEATING, seed=5000 + seed)

        torch.manual_seed(7000 + seed)
        staying = mean_returns(game, agent_for_everyone(agent_policy), BASELINE_EPISODES)

        rows = []
        for seat in range(n):
            for name, policy in SWITCHES:
                torch.manual_seed(8000 + seed)
                switching = mean_returns(game, one_seat_switches(seat, policy, agent_policy),
                                         SWITCH_EPISODES)[seat]
                gain, error, profitable = judge(staying[seat], switching)
                rows.append({"seat": seat, "switch": name, "staying": staying[seat],
                             "switching": switching, "gain": gain, "error": error,
                             "profitable": profitable})
        RESULTS[n][seed] = rows
        found = 0
        for row in rows:
            if row["profitable"]:
                found = found + 1
        print("  N=%-3d seed %d: %d of %d switches profitable   (%.1f min so far)"
              % (n, seed, found, len(rows), (time.time() - started_all) / 60.0), flush=True)

# %% [markdown]
# ## Every profitable switch, listed
#
# Each line names the seat, its strength, what it switched to, and what it gained. Seat 0 is the
# weakest player.

# %%
ACCURACIES = {}
for n in PLAYER_COUNTS:
    ACCURACIES[n] = sorted(accuracies_of(n))

any_found = False
for n in PLAYER_COUNTS:
    for seed in SEEDS:
        for row in RESULTS[n][seed]:
            if not row["profitable"]:
                continue
            any_found = True
            print("  N=%-3d seed %d  seat %2d (accuracy %.4f) switches to %-16s  %.4f -> %.4f  "
                  "gain %+.4f  (%.1f standard errors)"
                  % (n, seed, row["seat"], ACCURACIES[n][row["seat"]], row["switch"],
                     row["staying"], row["switching"], row["gain"], row["gain"] / row["error"]))
if not any_found:
    print("  no switch was profitable at any player count or seed")

# %% [markdown]
# ## The largest gain each seat could find
#
# Profitable or not, this is the best any single switch did for each seat, averaged over seeds. A
# column that sits close to zero everywhere means the agent's policy is hard to improve on by
# switching to a simple rule; a clearly positive entry marks where it is not.

# %%
for n in PLAYER_COUNTS:
    print("  N=%d   best gain from switching, by seat (seat 0 = weakest), mean over %d seeds"
          % (n, len(SEEDS)))
    cells = []
    for seat in range(n):
        best_per_seed = []
        for seed in SEEDS:
            best = None
            for row in RESULTS[n][seed]:
                if row["seat"] == seat and (best is None or row["gain"] > best):
                    best = row["gain"]
            best_per_seed.append(best)
        cells.append("%+.4f" % (sum(best_per_seed) / len(best_per_seed)))
    print("    " + "  ".join(cells))
    print()

# %% [markdown]
# ## Summary
#
# Read off the measurements above rather than stated in advance.

# %%
print("  N    seed   profitable switches   of")
summary = {}
for n in PLAYER_COUNTS:
    summary[n] = []
    for seed in SEEDS:
        found = 0
        for row in RESULTS[n][seed]:
            if row["profitable"]:
                found = found + 1
        summary[n].append(found)
        print("  %-4d %-6d %-21d %d" % (n, seed, found, len(RESULTS[n][seed])))

print()
for n in PLAYER_COUNTS:
    stable_seeds = 0
    for found_in_seed in summary[n]:
        if found_in_seed == 0:
            stable_seeds = stable_seeds + 1
    if stable_seeds == len(SEEDS):
        print("  N=%d: no seed's policy could be improved by any of the switches tried" % n)
    elif stable_seeds == 0:
        print("  N=%d: every seed's policy has at least one profitable switch, so none is an "
              "equilibrium" % n)
    else:
        print("  N=%d: %d of %d seeds survive every switch tried; the others are not equilibria"
              % (n, stable_seeds, len(SEEDS)))

# %% [markdown]
# ## What this does and does not establish
#
# A profitable switch is a **proof** that the policy is not an equilibrium: a player has a better
# option and would take it. No profitable switch is **not** proof that it is one, because an
# equilibrium has to resist every possible strategy and only four were tried. The one-sided
# strength of this test is the reason to state its result as a test rather than a score.
#
# This step also does not say whether the agent plays *well*, which is a different question. At
# five players the exact answer settles that; above eight nothing can.

# %% [markdown]
# ## Reproducibility
#
# The same switch measured twice from fresh seeds.

# %%
def measure_once():
    n = PLAYER_COUNTS[0]
    agent_policy = make_agent_policy(TRAINED[n][SEEDS[0]])
    game = GruelEnv(n=n, reward=REWARD, seating=SEATING, seed=9000)
    torch.manual_seed(9100)
    return mean_returns(game, one_seat_switches(0, shoot_weakest, agent_policy), 400)[0]


first = measure_once()
second = measure_once()
print("  first  run:", format(first, ".6f"))
print("  second run:", format(second, ".6f"))
print("  identical:", first == second)
