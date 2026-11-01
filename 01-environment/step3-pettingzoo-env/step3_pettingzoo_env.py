# %% [markdown]
# # Step 3 — The same game, wrapped in PettingZoo
#
# Step 2's game works. So why wrap it in a framework?
#
# Because the training code in folder 03 is not going to be written from scratch. It comes
# from CleanRL, and it expects an environment that looks a particular way: a standard method
# for resetting, a standard way of asking "whose turn is it", a standard place to find the
# reward. If our game speaks that language, existing training code can drive it.
#
# **PettingZoo** is the standard for multi-agent environments. The single-player equivalent
# is Gymnasium, which you may have seen with CartPole.
#
# The game underneath this notebook is *exactly* step 2's game. Same rules, same numbers.
# Everything new here is interface.

# %%
N = 3
SEED = 0

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import AECEnv

SHOOT_LEFT = 0
SHOOT_RIGHT = 1
MISS = 2
ACTION_NAMES = {SHOOT_LEFT: "shoot_left", SHOOT_RIGHT: "shoot_right", MISS: "miss"}

# %% [markdown]
# ## 3.1 AEC: why this game needs the awkward one
#
# PettingZoo has two styles:
#
# - **Parallel** — everybody acts at the same time. Good for something like a team game.
# - **AEC** (Agent Environment Cycle) — one agent acts, then the environment updates, then
#   the next agent acts.
#
# Our game is AEC, and not by choice. Only one player shoots per step, and *who* that is
# gets drawn at random. There is no round in which everybody acts. A parallel environment
# cannot express that.
#
# AEC costs us something. Because agents act one at a time and at irregular intervals,
# there is no single tidy sequence of "state, action, reward" for the whole game — each
# agent has its own scattered history. That is exactly what makes the training code in
# folder 03 hard, and it starts here.

# %% [markdown]
# ## 3.2 The four things PettingZoo requires
#
# | Thing | What it is |
# |---|---|
# | `possible_agents` | Every agent that could ever exist, as names. Never changes |
# | `agents` | Who is still in the game right now. Shrinks as players die |
# | `agent_selection` | Whose turn it is |
# | `terminations` / `truncations` | A dictionary per agent: is it finished? |
#
# The distinction between `terminations` and `truncations` is worth getting right:
#
# - **termination** — the game ended *by its own rules*. Someone won, or the game is stuck.
# - **truncation** — we cut it off from outside, because it hit a step limit.
#
# The payoff is the same either way. They are separated because a high truncation rate is
# a *bug signal*, not a result.

# %%
class GruelEnv(AECEnv):
    """Step 2's game, speaking PettingZoo."""

    metadata = {"name": "gruel_v0", "render_modes": ["ansi"], "is_parallelizable": False}

    def __init__(self, n=3, seating="random", seed=0):
        super().__init__()
        self.n = n
        self.seating = seating
        self.stagnation_rounds = 10
        self.max_steps = 200 * n
        self.render_mode = "ansi"

        self.rng = np.random.default_rng(seed)

        # Agents are named. Player i sits at position i, for the whole episode.
        self.possible_agents = []
        for i in range(n):
            self.possible_agents.append("player_" + str(i))

        # Two lookup tables so we can go between names and positions.
        self.agent_of_position = {}
        self.position_of_agent = {}
        for i in range(n):
            name = self.possible_agents[i]
            self.agent_of_position[i] = name
            self.position_of_agent[name] = i

        # Every agent has the same shape of observation and the same three actions.
        self.observation_spaces = {}
        self.action_spaces = {}
        for name in self.possible_agents:
            self.observation_spaces[name] = spaces.Box(low=0.0, high=1.0, shape=(4,),
                                                       dtype=np.float32)
            self.action_spaces[name] = spaces.Discrete(3)

    # PettingZoo asks for these as functions, not attributes.
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    # ---- reset ----

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.agents = list(self.possible_agents)

        accuracies_ascending = []
        for k in range(self.n):
            fraction = k / (self.n - 1)
            accuracies_ascending.append(0.4 + (0.8 - 0.4) * fraction)
        accuracies_ascending = np.array(accuracies_ascending)

        if self.seating == "ranked":
            self.accuracies = accuracies_ascending.copy()
        else:
            self.accuracies = self.rng.permutation(accuracies_ascending)

        self.order = list(range(self.n))
        self.t = 0
        self.miss_streak = 0
        self.elimination_order = []
        self.reason = None
        self.episode_over = False

        # The PettingZoo bookkeeping dictionaries, one entry per agent.
        self.rewards = {}
        self._cumulative_rewards = {}
        self.terminations = {}
        self.truncations = {}
        self.infos = {}
        for name in self.agents:
            self.rewards[name] = 0.0
            self._cumulative_rewards[name] = 0.0
            self.terminations[name] = False
            self.truncations[name] = False
            self.infos[name] = {}

        # Our own total, because _cumulative_rewards gets reset (see 3.4).
        self.episode_returns = {}
        for name in self.agents:
            self.episode_returns[name] = 0.0

        self.agent_selection = self.draw_actor()

    # ---- helpers ----

    def n_alive(self):
        return len(self.order)

    def draw_actor(self):
        position = int(self.rng.choice(self.order))
        return self.agent_of_position[position]

    def neighbours(self, position):
        index_in_order = self.order.index(position)
        how_many_alive = len(self.order)
        left = self.order[(index_in_order - 1) % how_many_alive]
        right = self.order[(index_in_order + 1) % how_many_alive]
        return left, right

    def stagnation_threshold(self):
        return self.stagnation_rounds * self.n_alive()

    def observe(self, agent):
        """The four numbers an agent gets to see. All zeros if it is dead."""
        position = self.position_of_agent[agent]

        if position not in self.order:
            return np.zeros(4, dtype=np.float32)

        left, right = self.neighbours(position)
        observation = [
            self.accuracies[position],
            self.accuracies[left],
            self.accuracies[right],
            self.n_alive() / self.n,
        ]
        return np.array(observation, dtype=np.float32)

    def target_of(self, agent, action):
        if action == MISS:
            return None
        position = self.position_of_agent[agent]
        left, right = self.neighbours(position)
        if action == SHOOT_LEFT:
            return left
        else:
            return right

    def add_reward(self, agent, amount):
        self.rewards[agent] = self.rewards[agent] + amount
        self.episode_returns[agent] = self.episode_returns[agent] + amount

    # ---- step ----

    def step(self, action):
        agent = self.agent_selection

        # A dead agent gets stepped once with None so PettingZoo can retire it.
        # This is the bit of the protocol that is easiest to get wrong. See 3.3.
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        # Clear last step's rewards before this step produces new ones.
        self._cumulative_rewards[agent] = 0
        self._clear_rewards()

        action = int(action)
        position = self.position_of_agent[agent]
        self.t = self.t + 1

        target = None
        victim = None
        hit = False

        if action == MISS:
            self.miss_streak = self.miss_streak + 1
        else:
            target = self.target_of(agent, action)
            self.miss_streak = 0
            roll = self.rng.random()
            hit = roll < self.accuracies[position]
            if hit:
                victim = target

        if victim is not None:
            victim_agent = self.agent_of_position[victim]
            self.terminations[victim_agent] = True
            self.order.remove(victim)
            self.elimination_order.append(victim)

        self.check_if_finished()

        for name in self.agents:
            self.infos[name] = {
                "t": self.t,
                "n_alive": self.n_alive(),
                "hit": hit if name == agent else None,
                "action": ACTION_NAMES[action] if name == agent else None,
                "termination_reason": self.reason,
            }

        if not self.episode_over:
            self.agent_selection = self.draw_actor()

        self._accumulate_rewards()
        # Put freshly dead agents at the front of the queue so they get retired.
        self._deads_step_first()

    def check_if_finished(self):
        if self.n_alive() == 1:
            self.reason = "sole_survivor"
            winner = self.agent_of_position[self.order[0]]
            self.add_reward(winner, 1.0)
            self.terminations[winner] = True

        elif self.miss_streak >= self.stagnation_threshold():
            self.reason = "stagnation"
            self.split_among_survivors()
            for position in self.order:
                self.terminations[self.agent_of_position[position]] = True

        elif self.t >= self.max_steps:
            self.reason = "hard_cap"
            self.split_among_survivors()
            for position in self.order:
                self.truncations[self.agent_of_position[position]] = True

        else:
            return

        self.episode_over = True

    def split_among_survivors(self):
        share = 1.0 / self.n_alive()
        for position in self.order:
            self.add_reward(self.agent_of_position[position], share)

    def render(self):
        parts = []
        for position in self.order:
            name = self.agent_of_position[position]
            parts.append(f"{name}(p={self.accuracies[position]:.2f})")
        ring = "  ".join(parts)
        return (f"t={self.t}  alive={self.n_alive()}/{self.n}  "
                f"miss_streak={self.miss_streak}/{self.stagnation_threshold()}\n"
                f"circle: {ring}\nactor:  {self.agent_selection}")

    def close(self):
        # Nothing to clean up for text rendering, but PettingZoo insists that any
        # environment defining render() also defines close(). See 3.6.
        pass


# %% [markdown]
# ## 3.3 The dead step — the one piece of the protocol that bites
#
# In PettingZoo, an agent does not vanish the moment it dies. The sequence is:
#
# 1. The agent is marked terminated.
# 2. It is still in `env.agents`, and it still gets selected once more.
# 3. Your loop must call `env.step(None)` for it.
# 4. *Then* PettingZoo removes it from `env.agents`.
#
# That extra step with `None` is the "dead step". `_was_dead_step` handles it, and
# `_deads_step_first()` makes sure dead agents get selected promptly rather than the episode
# carrying on around them.
#
# Why this matters practically: if you forget the `None`, your loop hangs or throws. If you
# treat that `None` step as a real decision, you feed garbage into training. **It fails
# quietly**, which is the worst kind.

# %% [markdown]
# ## 3.4 Two reward dictionaries, and why neither is the total
#
# PettingZoo gives you:
#
# - `rewards[agent]` — what this agent earned from the **most recent step only**. Cleared
#   constantly.
# - `_cumulative_rewards[agent]` — a running total, but it is **reset to 0 every time that
#   agent acts**, because that is when the value is handed to the learner.
#
# So neither one answers "what did this agent score this episode?". That is why the class
# above keeps its own `episode_returns` dictionary.

# %% [markdown]
# ## 3.5 The standard loop
#
# This is what every PettingZoo program looks like. `agent_iter()` keeps handing you
# whoever is next; `last()` gives you that agent's observation and reward.

# %%
def shoot_strongest(observation):
    left_accuracy = observation[1]
    right_accuracy = observation[2]
    if left_accuracy > right_accuracy:
        return SHOOT_LEFT
    else:
        return SHOOT_RIGHT


env = GruelEnv(n=N, seating="random", seed=SEED)
env.reset(seed=SEED)

print("accuracies by position:", np.round(env.accuracies, 2))
print()

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        # The dead step. No decision to make.
        env.step(None)
        continue

    action = shoot_strongest(observation)
    print(f"t={env.t + 1:>2}  {agent} plays {ACTION_NAMES[action]}", end="")

    env.step(action)

    if info is not None:
        pass
    print(f"   -> alive: {env.n_alive()}")

print()
print("ended because:", env.reason)
print("episode totals:", env.episode_returns)

# %% [markdown]
# ## 3.6 Does PettingZoo agree this is a valid environment?
#
# PettingZoo ships its own conformance test. This is not our test — it is the library
# checking that we implemented its protocol properly. It runs thousands of random steps and
# complains about anything inconsistent.
#
# Running it is worth more than any test I could write, because it knows the rules of the
# framework and I don't.
#
# It proved that on the first attempt. The class above originally had no `close()` method,
# and `api_test` refused it: *"If render method defined, then close method required."* A
# fair complaint — if an environment can open something, callers need a way to shut it. Our
# text rendering has nothing to close, so the method is empty, but the rule is right and it
# matters in step 5 where `render` opens a real window.

# %%
from pettingzoo.test import api_test

test_env = GruelEnv(n=N, seating="random", seed=SEED)
api_test(test_env, num_cycles=200, verbose_progress=False)
print()
print("api_test passed — this is a legitimate PettingZoo AEC environment.")

# %% [markdown]
# ## 3.7 The same numbers as step 2
#
# The wrapper must not have changed the game. Same policy, same seed count, and the
# survival probabilities should land on 8/27, 9/27, 10/27 again.

# %%
def run_many_pettingzoo(n, episodes, seed):
    env = GruelEnv(n=n, seating="random", seed=seed)

    accuracy_levels = []
    for k in range(n):
        fraction = k / (n - 1)
        accuracy_levels.append(0.4 + (0.8 - 0.4) * fraction)
    accuracy_levels = np.array(accuracy_levels)

    wins_per_level = np.zeros(n)

    for episode in range(episodes):
        env.reset()

        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            if termination or truncation:
                env.step(None)
                continue
            action = shoot_strongest(observation)
            env.step(action)

        for name in env.possible_agents:
            score = env.episode_returns[name]
            if score > 0:
                position = env.position_of_agent[name]
                accuracy_here = env.accuracies[position]
                differences = np.abs(accuracy_levels - accuracy_here)
                level_index = int(np.argmin(differences))
                wins_per_level[level_index] += score

    return accuracy_levels, wins_per_level / episodes


EPISODES = 50_000
levels, survival = run_many_pettingzoo(N, EPISODES, SEED)

print(f"{EPISODES:,} episodes through the PettingZoo wrapper")
print()
for i in range(len(levels)):
    print(f"  accuracy {levels[i]:.1f}  ->  survives {survival[i]:.4f}")
print()
print("adds to:", survival.sum())

if N == 3:
    exact_weakest_first = np.array([10 / 27, 9 / 27, 8 / 27])
    standard_error = np.sqrt(exact_weakest_first * (1 - exact_weakest_first) / EPISODES)
    print()
    print("standard errors away from exact:")
    for i in range(3):
        difference = survival[i] - exact_weakest_first[i]
        print(f"  accuracy {levels[i]:.1f}: {difference / standard_error[i]:+.2f}")

# %% [markdown]
# ## 3.8 The seeding check
#
# One extra wrinkle here. `reset(seed=...)` **rebuilds** the generator, so passing a seed
# every episode would give you the same episode over and over. Passing it once at the start
# and then calling plain `reset()` lets the stream continue.

# %%
def three_episode_fingerprint(seed):
    env = GruelEnv(n=N, seed=seed)
    env.reset(seed=seed)
    winners = []
    for episode in range(3):
        if episode > 0:
            env.reset()
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            if termination or truncation:
                env.step(None)
                continue
            env.step(shoot_strongest(observation))
        winners.append(round(float(env.accuracies[env.order[0]]), 2))
    return winners


print("seed 0, run again:", three_episode_fingerprint(0), three_episode_fingerprint(0))
print("seed 1           :", three_episode_fingerprint(1))
print()
print("same seed reproduces:", three_episode_fingerprint(0) == three_episode_fingerprint(0))

# %% [markdown]
# ## What to take from step 3
#
# - The **rules did not change**. Everything here is interface, and the survival numbers
#   are the same as step 2's.
# - AEC is forced on us: one random player acts per step, so there is no parallel round.
# - The **dead step** — `env.step(None)` for a terminated agent — is mandatory and fails
#   quietly if you get it wrong.
# - Neither `rewards` nor `_cumulative_rewards` is an episode total; you keep your own.
# - `api_test` is free verification from people who know the protocol better than we do.
#
# Next: `step4_heuristics.py`, which builds the four comparison policies and measures them
# properly.
