"""Step 8: train and measure both critics at 10 and 15 players.

A script and not a notebook because it runs for longer than an hour sequentially. It runs
every (player count, critic, seed) job in parallel, writes one JSON file per job into
results/, and saves each trained model and its record into models/. The notebook in this folder reads those files and does all the analysis.

    ../../03-agent-architecture/.venv/bin/python train_and_measure.py            # every job
    ../../03-agent-architecture/.venv/bin/python train_and_measure.py --workers 4
    ../../03-agent-architecture/.venv/bin/python train_and_measure.py --n 10 --arm local --seed 0

A job whose result file and model record both exist is skipped, so an interrupted run can be resumed.
Every job seeds itself, so any one of them reproduces on its own.
"""

import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor

import gymnasium
import numpy as np
import pettingzoo
import torch
import torch.nn as nn

from gruel_env import GruelEnv, MISS, SHOOT_LEFT, SHOOT_RIGHT, N_ACTIONS

PLAYER_COUNTS = [10, 15]
ARMS = ["centralised", "local"]
SEEDS = [0, 1, 2, 3, 4]
REWARD = "terminal"
SEATING = "ranked"

OBSERVATION_SIZE = 4
HIDDEN_SIZE = 64

UPDATES = 2000
EPISODES_PER_UPDATE = 96
EVALUATION_EPISODES = 10000

LEARNING_RATE = 0.003
CLIP = 0.2
EPOCHS = 4
MINIBATCH_SIZE = 256
VALUE_COEFFICIENT = 0.5
ENTROPY_COEFFICIENT = 0.01

# The switch test, as in step 2: each seat's return when everyone plays the agent, then when
# that seat alone switches to a simple rule.
BASELINE_EPISODES = 10000
SWITCH_EPISODES = 5000
STANDARD_ERRORS = 3

CRITIC_EPISODES = 200

RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ---- the agent: deliverable 03 step 10's, with the critic's input as the one thing that varies ----

class SharedActorCritic(nn.Module):
    def __init__(self, n_players, critic_view):
        super().__init__()
        self.n_players = n_players
        self.critic_view = critic_view
        # The actor is built first, so the same seed gives both arms the same actor.
        self.actor_first = nn.Linear(OBSERVATION_SIZE, HIDDEN_SIZE)
        self.actor_second = nn.Linear(HIDDEN_SIZE, N_ACTIONS)
        if critic_view == "centralised":
            critic_input_size = 2 * n_players + n_players
        elif critic_view == "local":
            critic_input_size = OBSERVATION_SIZE
        else:
            raise ValueError("critic_view must be 'centralised' or 'local'")
        self.critic_first = nn.Linear(critic_input_size, HIDDEN_SIZE)
        self.critic_second = nn.Linear(HIDDEN_SIZE, 1)
        self.activation = nn.Tanh()

    def action_logits(self, observation):
        hidden = self.activation(self.actor_first(observation))
        return self.actor_second(hidden)

    def value(self, critic_input):
        hidden = self.activation(self.critic_first(critic_input))
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


def critic_input_for(game, actor, observation, critic_view):
    """What the critic is shown. The ONLY place the two arms differ."""
    if critic_view == "centralised":
        state_array = global_state(game)
        agent_array = one_hot(actor, game.n)
        return np.concatenate([state_array, agent_array])
    return np.array(observation, dtype=np.float32)


def collect_batch(model, game, episodes):
    observations = {p: [] for p in range(game.n)}
    actions = {p: [] for p in range(game.n)}
    old_log_probabilities = {p: [] for p in range(game.n)}
    values = {p: [] for p in range(game.n)}
    returns = {p: [] for p in range(game.n)}
    critic_inputs = {p: [] for p in range(game.n)}

    for _ in range(episodes):
        game.reset()
        episode_observations = {p: [] for p in range(game.n)}
        episode_actions = {p: [] for p in range(game.n)}
        episode_log_probabilities = {p: [] for p in range(game.n)}
        episode_values = {p: [] for p in range(game.n)}
        episode_critic_inputs = {p: [] for p in range(game.n)}
        episode_banked_before = {p: [] for p in range(game.n)}

        for acting_agent in game.agent_iter():
            observation, reward, termination, truncation, info = game.last()
            if termination or truncation:
                game.step(None)
                continue

            actor = game.position_of_agent[acting_agent]
            banked_so_far = float(game.episode_returns[acting_agent])
            observation_tensor = torch.from_numpy(observation)
            critic_array = critic_input_for(game, actor, observation, model.critic_view)

            with torch.no_grad():
                logits = model.action_logits(observation_tensor)
                value = model.value(torch.from_numpy(critic_array))

            distribution = torch.distributions.Categorical(logits=logits)
            action = distribution.sample()

            episode_observations[actor].append(observation)
            episode_actions[actor].append(int(action))
            episode_log_probabilities[actor].append(float(distribution.log_prob(action)))
            episode_values[actor].append(float(value))
            episode_critic_inputs[actor].append(critic_array)
            episode_banked_before[actor].append(banked_so_far)

            game.step(int(action))

        for position in range(game.n):
            # The return for a decision is everything this player banked after it.
            name = game.agent_of_position[position]
            final_total = float(game.episode_returns[name])
            for index in range(len(episode_actions[position])):
                own_return = final_total - episode_banked_before[position][index]
                observations[position].append(episode_observations[position][index])
                actions[position].append(episode_actions[position][index])
                old_log_probabilities[position].append(
                    episode_log_probabilities[position][index])
                values[position].append(episode_values[position][index])
                critic_inputs[position].append(episode_critic_inputs[position][index])
                returns[position].append(own_return)

    flat_observations = []
    flat_actions = []
    flat_old_log_probabilities = []
    flat_values = []
    flat_returns = []
    flat_critic_inputs = []
    for position in range(game.n):
        flat_observations.extend(observations[position])
        flat_actions.extend(actions[position])
        flat_old_log_probabilities.extend(old_log_probabilities[position])
        flat_values.extend(values[position])
        flat_returns.extend(returns[position])
        flat_critic_inputs.extend(critic_inputs[position])

    batch = {
        "observations": torch.tensor(np.array(flat_observations), dtype=torch.float32),
        "actions": torch.tensor(flat_actions, dtype=torch.long),
        "old_log_probabilities": torch.tensor(flat_old_log_probabilities,
                                              dtype=torch.float32),
        "values": torch.tensor(flat_values, dtype=torch.float32),
        "returns": torch.tensor(flat_returns, dtype=torch.float32),
        "critic_inputs": torch.tensor(np.array(flat_critic_inputs), dtype=torch.float32),
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

            predicted_value = model.value(batch["critic_inputs"][index])
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


def train_one(n, arm, seed):
    torch.manual_seed(2000 + seed)
    model = SharedActorCritic(n_players=n, critic_view=arm)
    game = GruelEnv(n=n, reward=REWARD, seating=SEATING, seed=1000 + seed)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    update_rng = np.random.default_rng(seed)

    for update in range(UPDATES):
        batch = collect_batch(model, game, EPISODES_PER_UPDATE)
        ppo_update(model, optimiser, batch, update_rng)
    return model


# ---- measurement: folder 04 step 1's ----

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


def accuracies_of(n):
    game = GruelEnv(n=n, reward=REWARD, seating=SEATING, seed=0)
    game.reset(seed=0)
    return [round(float(a), 4) for a in game.accuracies]


def summarise(records, n):
    """Everything the notebook needs from 10,000 episodes, without keeping every decision."""
    ranked = sorted(accuracies_of(n))

    wins = {}
    for accuracy in ranked:
        wins[accuracy] = 0
    stagnated = 0
    capped = 0
    for record in records:
        if record["survivor_accuracy"] is not None:
            wins[record["survivor_accuracy"]] = wins[record["survivor_accuracy"]] + 1
        if record["stagnated"]:
            stagnated = stagnated + 1
        if record["hit_cap"]:
            capped = capped + 1

    stronger_by_rank = [0] * n
    eligible_by_rank = [0] * n
    stronger_by_alive = {}
    eligible_by_alive = {}
    misses = 0
    decisions = 0
    for record in records:
        for decision in record["decisions"]:
            decisions = decisions + 1
            if decision["action"] == MISS:
                misses = misses + 1
            if not eligible(decision):
                continue
            rank = ranked.index(decision["accuracy"])
            alive = str(decision["n_alive"])
            eligible_by_rank[rank] = eligible_by_rank[rank] + 1
            eligible_by_alive[alive] = eligible_by_alive.get(alive, 0) + 1
            if chose_stronger(decision):
                stronger_by_rank[rank] = stronger_by_rank[rank] + 1
                stronger_by_alive[alive] = stronger_by_alive.get(alive, 0) + 1

    targeting_by_rank = []
    for rank in range(n):
        targeting_by_rank.append(stronger_by_rank[rank] / eligible_by_rank[rank])
    targeting_by_alive = {}
    for alive in eligible_by_alive:
        targeting_by_alive[alive] = stronger_by_alive.get(alive, 0) / eligible_by_alive[alive]

    survival = []
    for accuracy in ranked:
        survival.append(wins[accuracy] / len(records))

    return {
        "episodes": len(records),
        "survival_by_rank": survival,
        "stagnation_rate": stagnated / len(records),
        "hard_cap_rate": capped / len(records),
        "abstention_rate": misses / decisions,
        "targeting_by_rank": targeting_by_rank,
        "targeting_by_alive": targeting_by_alive,
    }


# ---- the switch test: folder 04 step 2's, with every measurement seeded on its own ----

def shoot_strongest(game, actor, rng):
    left, right = game.neighbours(actor)
    if game.accuracies[left] >= game.accuracies[right]:
        return SHOOT_LEFT
    return SHOOT_RIGHT


def shoot_weakest(game, actor, rng):
    left, right = game.neighbours(actor)
    if game.accuracies[left] < game.accuracies[right]:
        return SHOOT_LEFT
    return SHOOT_RIGHT


def coin_flip(game, actor, rng):
    if rng.random() < 0.5:
        return SHOOT_LEFT
    return SHOOT_RIGHT


def always_miss(game, actor, rng):
    return MISS


SWITCHES = [("shoot_strongest", shoot_strongest), ("shoot_weakest", shoot_weakest),
            ("coin_flip", coin_flip), ("always_miss", always_miss)]


def mean_returns(n, agent_policy, switching_seat, switch_policy, episodes, seed):
    """Every seat's average return; one seat optionally plays a simple rule instead.

    A fresh game, a fresh rule stream and a reset torch stream for every measurement, so
    each number reproduces on its own whatever was measured before it.
    """
    game = GruelEnv(n=n, reward=REWARD, seating=SEATING, seed=seed)
    rule_rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    totals = [0.0] * n
    for _ in range(episodes):
        game.reset()
        for acting_agent in game.agent_iter():
            observation, reward, termination, truncation, info = game.last()
            if termination or truncation:
                game.step(None)
                continue
            actor = game.position_of_agent[acting_agent]
            if actor == switching_seat:
                action = switch_policy(game, actor, rule_rng)
            else:
                action = agent_policy(game, actor)
            game.step(action)

        for position in range(n):
            name = game.agent_of_position[position]
            totals[position] = totals[position] + game.episode_returns[name]

    averages = []
    for position in range(n):
        averages.append(totals[position] / episodes)
    return averages


def standard_error(probability, episodes):
    probability = min(max(probability, 0.0), 1.0)
    return (probability * (1 - probability) / episodes) ** 0.5


def switch_test(n, model, seed):
    agent_policy = make_agent_policy(model)
    staying = mean_returns(n, agent_policy, None, None, BASELINE_EPISODES, 11000 + seed)
    rows = []
    for seat in range(n):
        for switch_index in range(len(SWITCHES)):
            name, rule = SWITCHES[switch_index]
            measurement_seed = 12000 + 1000 * seed + 10 * seat + switch_index
            switching = mean_returns(n, agent_policy, seat, rule, SWITCH_EPISODES,
                                     measurement_seed)[seat]
            gain = switching - staying[seat]
            error = (standard_error(staying[seat], BASELINE_EPISODES) ** 2
                     + standard_error(switching, SWITCH_EPISODES) ** 2) ** 0.5
            rows.append({
                "seat": seat,
                "switch": name,
                "staying": staying[seat],
                "switching": switching,
                "gain": gain,
                "error": error,
                "profitable": gain > STANDARD_ERRORS * error,
            })
    return rows


def explained_variance(model, n, seed):
    torch.manual_seed(8000 + seed)
    game = GruelEnv(n=n, reward=REWARD, seating=SEATING, seed=8000 + seed)
    batch = collect_batch(model, game, CRITIC_EPISODES)
    residual = batch["returns"] - batch["values"]
    return 1.0 - float(residual.var()) / float(batch["returns"].var())


# ---- saving the trained model and its record ----
# The same code as every training notebook in this repository.


MODELS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
NOTEBOOK = "train_and_measure.py"
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
                                check=True, cwd=os.path.dirname(os.path.abspath(__file__)))
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
        "step": os.path.basename(os.path.dirname(os.path.abspath(__file__))),
        "deliverable": os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
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
            "notebook_source_sha256": file_sha256(os.path.abspath(__file__)),
            "gruel_env_sha256": file_sha256(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gruel_env.py")),
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


# ---- one job ----

def result_path(n, arm, seed):
    return os.path.join(RESULTS_FOLDER, "n%d_%s_seed%d.json" % (n, arm, seed))


def run_job(job):
    n, arm, seed = job
    torch.set_num_threads(1)
    path = result_path(n, arm, seed)
    key = run_key(n, SEATING, REWARD, arm, UPDATES, seed)
    model_record = os.path.join(MODELS_FOLDER, key + ".json")
    if os.path.exists(path) and os.path.exists(model_record):
        return path, "skipped, already done"

    started = time.time()
    model = train_one(n, arm, seed)
    trained_minutes = (time.time() - started) / 60.0
    save_model(model, key, mappo_settings(n, REWARD, SEATING, seed, UPDATES, EPISODES_PER_UPDATE,
                                          arm, time.time() - started, "training only"))

    # Measured exactly as folder 04 step 1 measures, so the centralised arm can be checked
    # against it seed for seed.
    game = GruelEnv(n=n, reward=REWARD, seating=SEATING, seed=5000 + seed)
    torch.manual_seed(7000 + seed)
    records = play_episodes(game, make_agent_policy(model), EVALUATION_EPISODES)
    result = summarise(records, n)

    result["switch_test"] = switch_test(n, model, seed)
    result["explained_variance"] = explained_variance(model, n, seed)

    result["n"] = n
    result["arm"] = arm
    result["seed"] = seed
    result["accuracies"] = sorted(accuracies_of(n))
    result["training_minutes"] = trained_minutes
    result["total_minutes"] = (time.time() - started) / 60.0
    result["torch"] = torch.__version__
    result["python"] = platform.python_version()

    temporary = path + ".part"
    with open(temporary, "w") as handle:
        json.dump(result, handle, indent=1)
    os.replace(temporary, path)
    return path, "done in %.1f min" % result["total_minutes"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--arm", default=None)
    parser.add_argument("--seed", type=int, default=None)
    arguments = parser.parse_args()

    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    jobs = []
    for n in PLAYER_COUNTS:
        for arm in ARMS:
            for seed in SEEDS:
                if arguments.n is not None and n != arguments.n:
                    continue
                if arguments.arm is not None and arm != arguments.arm:
                    continue
                if arguments.seed is not None and seed != arguments.seed:
                    continue
                jobs.append((n, arm, seed))
    # The longest jobs first, so the pool finishes evenly.
    jobs.sort(key=lambda job: -job[0])

    print("jobs:", len(jobs), " workers:", arguments.workers, flush=True)
    started = time.time()
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        futures = []
        for job in jobs:
            futures.append((job, pool.submit(run_job, job)))
        for job, future in futures:
            path, status = future.result()
            print("  N=%-3d %-11s seed %d  %s   (%.1f min since start)"
                  % (job[0], job[1], job[2], status, (time.time() - started) / 60.0), flush=True)
    print("all done in %.1f min" % ((time.time() - started) / 60.0), flush=True)


if __name__ == "__main__":
    main()
