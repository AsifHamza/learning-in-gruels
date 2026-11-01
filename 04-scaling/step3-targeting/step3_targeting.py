# %% [markdown]
# # Step 3 — Who the agent shoots
#
# Steps 1 and 2 look at **outcomes**: who survives, and whether a player could do better by
# switching. Outcomes cannot say what a policy does, because two different policies can produce
# the same survival numbers. This step measures behaviour directly.
#
# Every time a player fires, it fires at one of its two neighbours. The question is: **how often
# does it pick the stronger one?** Read by the player's own strength and by how many players are
# left, that one number separates policies that survival cannot.
#
# Five players is included because folder 02 knows the exact equilibrium's targeting there, so the
# measurement is checked before it is read at 10 and 15.

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
# The agent is trained exactly as in step 1: same code, budget and seeds.

# %%
PLAYER_COUNTS = [5, 10, 15]
REWARD = "terminal"
SEATING = "ranked"

OBSERVATION_SIZE = 4
HIDDEN_SIZE = 64

UPDATES = 2000
EPISODES_PER_UPDATE = 96
EVALUATION_EPISODES = 10000
REFERENCE_EPISODES = 10000
SEEDS = [0, 1, 2]

LEARNING_RATE = 0.003
CLIP = 0.2
EPOCHS = 4
MINIBATCH_SIZE = 256
VALUE_COEFFICIENT = 0.5
ENTROPY_COEFFICIENT = 0.01

MATCH_THRESHOLD = 0.10

print("player counts:", PLAYER_COUNTS, " seeds:", SEEDS)
print("a reference is named as a match only if it is within %.2f on average" % MATCH_THRESHOLD)

# %% [markdown]
# ## What counts as a targeting decision
#
# A decision only counts if the question has an answer. Three cases are excluded:
#
# 1. **The player abstained.** It chose no target.
# 2. **Two players are left.** Both neighbours are the same opponent.
# 3. **The two neighbours are equally strong.** There is no stronger one.
#
# When nothing is left to count, the rate is reported as missing, not as zero. Zero would claim the
# player always shot the weaker neighbour, which is the opposite of "it never chose".

# %% [markdown]
# ## The exact answer at five players
#
# From folder 02, step 5, with players seated weakest to strongest, weighted per decision so it
# is the same quantity as the agent's targeting measured here. Only the two players at the
# ends of the ranking fire at their stronger neighbour every time; the three in the middle mostly
# fire at their *weaker* one. And the more players have been eliminated, the more often the
# stronger neighbour is chosen.

# %%
EXACT_TARGETING_N5_BY_RANK = [1.00000, 0.39790, 0.31072, 0.37137, 1.00000]
EXACT_TARGETING_N5_BY_ALIVE = {5: 0.40, 4: 0.50, 3: 1.00}

print("  exact, by rank (0 = weakest):", "  ".join(format(v, ".4f") for v in EXACT_TARGETING_N5_BY_RANK))
print("  exact, by players left:     ", "  ".join("%d alive %.2f" % (k, EXACT_TARGETING_N5_BY_ALIVE[k])
                                                  for k in sorted(EXACT_TARGETING_N5_BY_ALIVE, reverse=True)))

# %% [markdown]
# ## Four patterns to read the agent against
#
# | Pattern | The ends of the ranking | The middle | Depends on players left? |
# |---|---|---|---|
# | matches the exact equilibrium | about 1.0 | about 0.3 to 0.4 at N=5 | strongly |
# | copies `next_weaker` | 1.0 | low, and lower as N grows | yes, as 2 in k |
# | copies `shoot_strongest` | high | high | no |
# | shoots at random | about 0.5 | about 0.5 | no |
#
# `next_weaker` is Archetti's rule projected onto the two neighbours, the same projection as
# folder 02 step 5.5: shoot the strongest neighbour who is still weaker than you, and if both are
# stronger, shoot the stronger one. **It matters because it is a simple fixed rule that already has
# the U-shape.** Under `ranked` seating the circle stays sorted as players die, so the
# immediately weaker player is always the left-hand neighbour, and the weakest player's left-hand
# neighbour is the strongest. `next_weaker` is therefore the same as "always shoot left" here, and it
# shoots the stronger neighbour at exactly two of the $k$ seats when $k$ players are left: the
# weakest and the strongest. Any agent that mostly fires one way round the circle will look U-shaped
# by rank under this seating, whatever it has learned about strength.
#
# The middle column separates the equilibrium from `shoot_strongest` and `coin_flip`. Only its
# *size* separates the equilibrium from `next_weaker`, and only where the exact answer is known.

# %% [markdown]
# ## The agent, its training and the targeting measurement, copied forward
#
# The agent and its training are step 1's, unchanged. The targeting functions are deliverable
# 03's, unchanged. Rule 1 says a step copies earlier code forward rather than importing it, so
# this folder carries its own copy and can be handed over on its own.

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


def play_episodes(game, policy, episodes):
    records = []
    for _ in range(episodes):
        game.reset()
        decisions = []

        for acting_agent in game.agent_iter():
            observation, reward, termination, truncation, info = game.last()
            if termination or truncation:
                game.step(None)
                continue

            actor = game.position_of_agent[acting_agent]
            action = policy(game, actor)

            # Recorded BEFORE stepping, because a kill contracts the circle.
            left, right = game.neighbours(actor)
            decisions.append({
                "accuracy": round(float(game.accuracies[actor]), 4),
                "action": action,
                "left_accuracy": round(float(game.accuracies[left]), 4),
                "right_accuracy": round(float(game.accuracies[right]), 4),
                "n_alive": game.n_alive(),
            })

            game.step(action)

        if len(game.order) == 1:
            survivor_accuracy = round(float(game.accuracies[game.order[0]]), 4)
        else:
            survivor_accuracy = None

        records.append({
            "survivor_accuracy": survivor_accuracy,
            "stagnated": game.reason == "stagnation",
            "hit_cap": game.reason == "hard_cap",
            "decisions": decisions,
        })
    return records


def eligible(decision):
    if decision["action"] == MISS:
        return False        # never fired, so it selected no target
    if decision["n_alive"] <= 2:
        return False        # left and right are the same player
    if decision["left_accuracy"] == decision["right_accuracy"]:
        return False        # neither neighbour is the stronger one
    return True


def chose_stronger(decision):
    if decision["action"] == SHOOT_LEFT:
        chosen = decision["left_accuracy"]
        other = decision["right_accuracy"]
    else:
        chosen = decision["right_accuracy"]
        other = decision["left_accuracy"]
    return chosen > other


def targeting_rate(records, accuracy=None):
    stronger = 0
    decisions = 0
    for record in records:
        for decision in record["decisions"]:
            if accuracy is not None and decision["accuracy"] != accuracy:
                continue
            if not eligible(decision):
                continue
            decisions = decisions + 1
            if chose_stronger(decision):
                stronger = stronger + 1
    if decisions == 0:
        return None, 0, 0
    return stronger / decisions, stronger, decisions


def shoot_strongest(game, actor):
    left, right = game.neighbours(actor)
    if game.accuracies[left] >= game.accuracies[right]:
        return SHOOT_LEFT
    return SHOOT_RIGHT


def next_weaker(game, actor):
    """Archetti's rule on the circle: the strongest neighbour still weaker than you, else the stronger."""
    left, right = game.neighbours(actor)
    mine = game.accuracies[actor]
    pool = []
    for candidate in (left, right):
        if game.accuracies[candidate] < mine:
            pool.append(candidate)
    if len(pool) == 0:
        pool = [left, right]      # both outgun you: you are the weak one, so fire at the stronger
    target = pool[0]
    for candidate in pool:
        if game.accuracies[candidate] > game.accuracies[target]:
            target = candidate
    if target == left:
        return SHOOT_LEFT
    return SHOOT_RIGHT


def coin_flip(game, actor):
    if reference_rng.random() < 0.5:
        return SHOOT_LEFT
    return SHOOT_RIGHT


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


def measure(model, n, seed):
    game = GruelEnv(n=n, reward=REWARD, seating=SEATING, seed=5000 + seed)
    torch.manual_seed(7000 + seed)
    return play_episodes(game, make_agent_policy(model), EVALUATION_EPISODES)


def targeting_by_alive(records):
    stronger_count = {}
    decision_count = {}
    for record in records:
        for decision in record["decisions"]:
            if not eligible(decision):
                continue
            alive_then = decision["n_alive"]
            if alive_then not in decision_count:
                stronger_count[alive_then] = 0
                decision_count[alive_then] = 0
            decision_count[alive_then] = decision_count[alive_then] + 1
            if chose_stronger(decision):
                stronger_count[alive_then] = stronger_count[alive_then] + 1
    rows = {}
    for alive in sorted(decision_count):
        rows[alive] = (stronger_count[alive] / decision_count[alive],
                       stronger_count[alive],
                       decision_count[alive])
    return rows


def targeting_profile(records, accuracies):
    profile = {}
    for accuracy in accuracies:
        rate, stronger, decisions = targeting_rate(records, accuracy=accuracy)
        profile[accuracy] = rate
    return profile

# %%
reference_rng = np.random.default_rng(0)

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
NOTEBOOK = "step3_targeting.ipynb"
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
# ## Training and measuring
#
# One model per player count per seed, then 10,000 measured episodes each. Every decision is
# recorded with the shooter's accuracy, both neighbours' accuracies and the number of players left,
# captured *before* the shot, because a kill changes who the neighbours are.

# %%
TRAINED = {}
RECORDS = {}
started_all = time.time()
for n in PLAYER_COUNTS:
    TRAINED[n] = {}
    RECORDS[n] = {}
    for seed in SEEDS:
        started = time.time()
        TRAINED[n][seed] = train_one(n, seed)
        save_model(TRAINED[n][seed], run_key(n, SEATING, REWARD, "centralised", UPDATES, seed),
                   mappo_settings(n, REWARD, SEATING, seed, UPDATES, EPISODES_PER_UPDATE,
                                  "centralised", time.time() - started, "training only"))
        RECORDS[n][seed] = measure(TRAINED[n][seed], n, seed)
        print("  N=%-3d seed %d trained and measured in %4.1f min"
              % (n, seed, (time.time() - started) / 60.0), flush=True)
print("  total %.1f min" % ((time.time() - started_all) / 60.0))

# %% [markdown]
# ## The references, measured in the same game
#
# No training needed: these are fixed rules, played under the same seating and the same measurement.
# `next_weaker` is deterministic and `coin_flip` draws from its own stream, so adding a reference
# does not move any random number the agent sees.

# %%
REFERENCES = {}
for n in PLAYER_COUNTS:
    REFERENCES[n] = {}
    for label, policy in [("shoot_strongest", shoot_strongest), ("next_weaker", next_weaker),
                          ("coin_flip", coin_flip)]:
        game = GruelEnv(n=n, reward=REWARD, seating=SEATING, seed=7000 + n)
        REFERENCES[n][label] = play_episodes(game, policy, REFERENCE_EPISODES)
print("  references measured at", PLAYER_COUNTS)

# %% [markdown]
# ## Targeting by strength
#
# For each rank, the share of eligible decisions where the player shot its stronger neighbour.
# The three seeds are pooled for the agent row, and shown separately below it so the spread is
# visible.

# %%
def by_rank(records, n):
    ranked = sorted(accuracies_of(n))
    profile = targeting_profile(records, ranked)
    values = []
    for accuracy in ranked:
        values.append(profile[accuracy])
    return values


def shown(value):
    if value is None:
        return "  none"
    return "%.4f" % value


def pooled(n):
    combined = []
    for seed in SEEDS:
        combined.extend(RECORDS[n][seed])
    return combined


AGENT_BY_RANK = {}
REFERENCE_BY_RANK = {}
for n in PLAYER_COUNTS:
    AGENT_BY_RANK[n] = by_rank(pooled(n), n)
    REFERENCE_BY_RANK[n] = {}
    for label in REFERENCES[n]:
        REFERENCE_BY_RANK[n][label] = by_rank(REFERENCES[n][label], n)

    print("  N=%d, rank 0 = weakest" % n)
    print("    %-16s %s" % ("agent, pooled", " ".join(shown(v) for v in AGENT_BY_RANK[n])))
    for seed in SEEDS:
        print("    %-16s %s" % ("  seed %d" % seed, " ".join(shown(v) for v in by_rank(RECORDS[n][seed], n))))
    for label in REFERENCE_BY_RANK[n]:
        print("    %-16s %s" % (label, " ".join(shown(v) for v in REFERENCE_BY_RANK[n][label])))
    if n == 5:
        print("    %-16s %s" % ("exact", " ".join(shown(v) for v in EXACT_TARGETING_N5_BY_RANK)))
    print()

# %% [markdown]
# ## Which pattern, measured rather than labelled
#
# The distance to each reference is the average absolute difference across ranks. A reference is
# named only if it is within the match threshold; otherwise the answer is "neither", with the
# nearest distance shown. The exact equilibrium is a reference only where it is known.
#
# The U-shape test is deliberately strict: the *lower* of the two ends must beat the *highest*
# middle rank, so a shape that holds only on average does not count.

# %%
def distance(first, second):
    total = 0.0
    count = 0
    for a, b in zip(first, second):
        if a is None or b is None:
            continue
        total = total + abs(a - b)
        count = count + 1
    return total / count


def is_u_shaped(values):
    ends = min(values[0], values[-1])
    middle = max(values[1:-1])
    return ends > middle


VERDICT = {}
for n in PLAYER_COUNTS:
    candidates = {}
    for label in REFERENCE_BY_RANK[n]:
        candidates[label] = distance(AGENT_BY_RANK[n], REFERENCE_BY_RANK[n][label])
    if n == 5:
        candidates["exact equilibrium"] = distance(AGENT_BY_RANK[n], EXACT_TARGETING_N5_BY_RANK)
    nearest = min(candidates, key=lambda label: candidates[label])
    if candidates[nearest] <= MATCH_THRESHOLD:
        verdict = "matches " + nearest
    else:
        verdict = "neither (nearest is %s, %.4f away)" % (nearest, candidates[nearest])
    VERDICT[n] = verdict
    u = is_u_shaped(AGENT_BY_RANK[n])
    print("  N=%-3d U-shaped: %-5s  distances: %s" % (n, u, "  ".join(
        "%s %.4f" % (label, candidates[label]) for label in candidates)))
    print("         -> %s" % verdict)

# %% [markdown]
# ## Why a fixed direction draws the U
#
# The claim above is checked rather than asserted: under `ranked` seating every one of
# `next_weaker`'s shots should go left, and its share of shots at the stronger neighbour with $k$
# players left should be $2/k$, because only the weakest and the strongest of the $k$ have a
# stronger player on their left.

# %%
for n in PLAYER_COUNTS:
    left_shots = 0
    shots = 0
    for record in REFERENCES[n]["next_weaker"]:
        for decision in record["decisions"]:
            if decision["action"] == MISS:
                continue
            shots = shots + 1
            if decision["action"] == SHOOT_LEFT:
                left_shots = left_shots + 1
    rows = targeting_by_alive(REFERENCES[n]["next_weaker"])
    worst = 0.0
    for alive_count in rows:
        worst = max(worst, abs(rows[alive_count][0] - 2.0 / alive_count))
    print("  N=%-3d next_weaker shoots left in %.4f of its shots; largest gap from 2/k by players left %.4f"
          % (n, left_shots / shots, worst))

# %% [markdown]
# ## Targeting by players left
#
# The exact equilibrium's targeting changes as the circle contracts. A policy that ignores how many
# players are left cannot reproduce that.

# %%
AGENT_BY_ALIVE = {}
for n in PLAYER_COUNTS:
    AGENT_BY_ALIVE[n] = targeting_by_alive(pooled(n))
    print("  N=%d, agent pooled over seeds" % n)
    row = []
    for alive_count in sorted(AGENT_BY_ALIVE[n], reverse=True):
        rate, stronger, decisions = AGENT_BY_ALIVE[n][alive_count]
        row.append("%d left %.4f" % (alive_count, rate))
    print("    " + "   ".join(row))
    rates = [AGENT_BY_ALIVE[n][k][0] for k in sorted(AGENT_BY_ALIVE[n])]
    print("    range across players left: %.4f" % (max(rates) - min(rates)))
    reference_rows = targeting_by_alive(REFERENCES[n]["next_weaker"])
    gaps = []
    for alive_count in sorted(AGENT_BY_ALIVE[n], reverse=True):
        gaps.append("%d left %.4f" % (alive_count, abs(AGENT_BY_ALIVE[n][alive_count][0]
                                                         - reference_rows[alive_count][0])))
    print("    gap from next_weaker: " + "   ".join(gaps))
    print()

# %% [markdown]
# ## Checking the measurement at five players
#
# Where the exact answer is known, the agent should match it both ways: by strength and by players
# left. Deliverable 03 established that it does; this checks that this folder measures it the same
# way before the measurement is read where nothing can check it.

# %%
print("  by players left, agent against exact:")
worst_gap = 0.0
for alive_count in sorted(EXACT_TARGETING_N5_BY_ALIVE, reverse=True):
    measured = AGENT_BY_ALIVE[5][alive_count][0]
    exact = EXACT_TARGETING_N5_BY_ALIVE[alive_count]
    worst_gap = max(worst_gap, abs(measured - exact))
    print("    %d left   agent %.4f   exact %.2f   gap %.4f" % (alive_count, measured, exact, abs(measured - exact)))
print()
print("  worst gap by players left: %.4f" % worst_gap)
print("  verdict by strength:      ", VERDICT[5])
if worst_gap <= MATCH_THRESHOLD and VERDICT[5] == "matches exact equilibrium":
    print("  -> the measurement recovers the exact equilibrium where it is known")
else:
    print("  -> the measurement does NOT recover the exact equilibrium at N=5; read 10 and 15 with caution")

# %% [markdown]
# ## The upper-middle players, read against step 2
#
# Step 2 found that at 15 players the profitable switches were all made by strong players short of
# the strongest, and that most of them gained by switching to `shoot_strongest`. If that is right,
# those players should be firing at their stronger neighbour well below every time. This reads
# their targeting directly.

# %%
largest = PLAYER_COUNTS[-1]
ranked = sorted(accuracies_of(largest))
upper_middle = list(range(8, largest - 1))
print("  N=15, ranks %d to %d (the strongest is rank %d)" % (upper_middle[0], upper_middle[-1], largest - 1))
upper_values = []
for rank in upper_middle:
    agent_value = AGENT_BY_RANK[largest][rank]
    reference_value = REFERENCE_BY_RANK[largest]["shoot_strongest"][rank]
    upper_values.append(agent_value)
    print("    rank %2d  accuracy %.4f   agent %s   shoot_strongest %s"
          % (rank, ranked[rank], shown(agent_value), shown(reference_value)))
upper_mean = sum(upper_values) / len(upper_values)
print()
print("  agent's mean over those ranks: %.4f" % upper_mean)
if upper_mean < 0.9:
    print("  -> they fire at their stronger neighbour well below every time, consistent with step 2")
else:
    print("  -> they already fire at their stronger neighbour nearly every time; step 2's gains are not")
    print("     explained by targeting alone")

# %% [markdown]
# ## What this step hands to the next one
#
# A description of the agent's behaviour at each player count, measured in a way that recovers the
# exact answer where one exists. What it cannot say is whether behaviour at 10 and 15 players is the
# *same* behaviour that was checked at 5 and 7, or something new. Comparing like with like, at the
# same number of players left, is the last step.

# %% [markdown]
# ## Reproducibility
#
# Two random streams are reset, not one: the environment's, and torch's global stream, because a
# trained policy samples its actions.

# %%
def measure_once():
    first_n = PLAYER_COUNTS[0]
    torch.manual_seed(7100)
    game = GruelEnv(n=first_n, reward=REWARD, seating=SEATING, seed=9000)
    records = play_episodes(game, make_agent_policy(TRAINED[first_n][SEEDS[0]]), 300)
    rate, stronger, decisions = targeting_rate(records)
    return rate


first = measure_once()
second = measure_once()
print("  first  run:", format(first, ".6f"))
print("  second run:", format(second, ".6f"))
print("  identical:", first == second)
