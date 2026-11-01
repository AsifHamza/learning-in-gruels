# The gruel game as a PettingZoo environment.
#
# Folder 01 builds this up in three steps: the rules on paper, then a plain Python
# game, then this -- the same game speaking PettingZoo. This file is that third step,
# trimmed to what the training code needs: no rendering, no keyboard play.
#
# Every folder has its own copy. That duplication is deliberate, so any one folder
# can be handed over on its own.
#
# The rules, in one paragraph: N players sit on a circle, each with a fixed accuracy.
# One player is picked at random to act. They may shoot their left neighbour, shoot
# their right neighbour, or deliberately miss. A shot hits with probability equal to
# the shooter's accuracy. A player who is hit is removed and the circle contracts, so
# their two neighbours become adjacent. Last one alive wins.

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import AECEnv

# The three actions. Integers because a neural network outputs three numbers.
SHOOT_LEFT = 0
SHOOT_RIGHT = 1
MISS = 2

N_ACTIONS = 3

ACTION_NAMES = {SHOOT_LEFT: "shoot_left", SHOOT_RIGHT: "shoot_right", MISS: "miss"}

# The two reward profiles.
#
# `terminal` is the honest one: you get 1 for winning and nothing else. It is what
# every validation run uses, because it is the objective the maths solves.
#
# `shaped` adds hints to make learning easier. Note the cost of doing that: a player
# who kills everyone but then dies can bank more than a player who simply wins. The
# hint and the goal are not quite the same thing.
REWARD_PROFILES = {
    "terminal": {"win": 1.0, "eliminated": 0.0, "kill": 0.0, "survive": 0.0},
    "shaped": {"win": 1.0, "eliminated": -1.0, "kill": 0.1, "survive": 0.001},
}


def make_accuracies(n, lowest=0.4, highest=0.8):
    """Evenly spaced accuracies from lowest to highest, ascending.

    At n=3 this gives [0.4, 0.6, 0.8]. At n=2, [0.4, 0.8].
    """
    accuracies = []
    for k in range(n):
        if n == 1:
            fraction = 0.0
        else:
            fraction = k / (n - 1)
        accuracy = lowest + (highest - lowest) * fraction
        accuracies.append(accuracy)
    return np.array(accuracies)


class GruelEnv(AECEnv):
    """One gruel game, as a PettingZoo AEC environment.

    AEC means Agent Environment Cycle: one agent acts, the environment updates, then
    the next agent acts. This game has no choice about that -- only one player shoots
    per step, and who that is gets drawn at random, so there is no round in which
    everybody acts at once.

    Use it the way every PettingZoo program does:

        env.reset()
        for acting_agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            if termination or truncation:
                env.step(None)
                continue
            env.step(my_chosen_action)
    """

    metadata = {"name": "gruel_v0", "render_modes": ["ansi"], "is_parallelizable": False}

    def __init__(self, n, reward="terminal", seating="ranked", seed=0):
        super().__init__()

        if reward not in REWARD_PROFILES:
            raise ValueError("reward must be 'terminal' or 'shaped'")
        if seating not in ("ranked", "random"):
            raise ValueError("seating must be 'ranked' or 'random'")

        self.n = n
        self.reward_profile = reward
        self.reward_params = dict(REWARD_PROFILES[reward])
        self.seating = seating
        self.rng = np.random.default_rng(seed)
        self.render_mode = "ansi"

        # Stop an episode that will not end on its own. Everyone may abstain forever,
        # which is a legal strategy, so the game needs a cut-off.
        #
        # The limit SCALES DOWN as players die -- see stagnation_threshold(). Ten
        # rounds of a three-player game is far fewer steps than ten rounds of a
        # fifteen-player one, and the cut-off should mean the same thing at both.
        self.stagnation_rounds = 10
        self.max_steps = 200 * n

        # Agents are named. Player i sits at position i, for the whole episode.
        self.possible_agents = []
        for i in range(n):
            name = "player_" + str(i)
            self.possible_agents.append(name)

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
            self.observation_spaces[name] = spaces.Box(
                low=0.0, high=1.0, shape=(4,), dtype=np.float32
            )
            self.action_spaces[name] = spaces.Discrete(N_ACTIONS)

        self.agents = []

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

        accuracies_ascending = make_accuracies(self.n)
        if self.seating == "ranked":
            # Weakest at position 0, ascending clockwise. Fixed, so every episode has
            # the same layout. The two best shots are always adjacent, so they shoot
            # each other and the weakest player is protected.
            self.accuracies = accuracies_ascending.copy()
        else:
            # A fresh shuffle each episode, so that protection becomes a lottery.
            self.accuracies = self.rng.permutation(accuracies_ascending)

        # order holds the positions still in the game, in circle order. Removing a
        # position from this list IS the contraction -- the two neighbours become
        # adjacent automatically, because they become adjacent in the list.
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

        # Our own running total. _cumulative_rewards gets zeroed every time an agent
        # acts, so it cannot be used for the episode total.
        self.episode_returns = {}
        for name in self.agents:
            self.episode_returns[name] = 0.0

        self.agent_selection = self.draw_actor()

    # ---- helpers ----

    def n_alive(self):
        return len(self.order)

    def draw_actor(self):
        """Pick who acts, uniformly at random from the players still alive.

        With replacement: the same player can act twice in a row. This is the
        random-order variant, and it is why one agent may make eight decisions in an
        episode while another makes one.
        """
        position = int(self.rng.choice(self.order))
        return self.agent_of_position[position]

    def neighbours(self, position):
        """The player to the left and to the right of a position on the circle.

        With only two left, both answers are the same opponent.
        """
        index_in_order = self.order.index(position)
        how_many_alive = len(self.order)
        left = self.order[(index_in_order - 1) % how_many_alive]
        right = self.order[(index_in_order + 1) % how_many_alive]
        return left, right

    def stagnation_threshold(self):
        """The stuck-limit. Scales down as players die."""
        return self.stagnation_rounds * self.n_alive()

    def observe(self, agent):
        """What one player sees: four numbers. All zeros if it is dead.

        Its own accuracy, its two neighbours' accuracies, and how much of the field is
        left. Note what is NOT here: the player's own position, the full seating, or
        anything about the other players further round the circle. An agent cannot
        tell where it sits in the global ranking.
        """
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
        """Which position an action points at, or None for a deliberate miss."""
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

        # A finished agent gets stepped once with None so PettingZoo can retire it.
        # This is the part of the protocol that is easiest to get wrong.
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        # Clear last step's rewards before this step produces new ones.
        self._cumulative_rewards[agent] = 0
        self._clear_rewards()

        action = int(action)
        if action not in ACTION_NAMES:
            raise ValueError("action must be 0, 1 or 2")

        position = self.position_of_agent[agent]
        self.t = self.t + 1

        victim = None
        hit = False

        # Work out the target BEFORE anyone is removed. Reading neighbours after a
        # removal gives the wrong answer -- the circle has already contracted.
        if action == MISS:
            # A deliberate miss. Nothing is fired, so the streak grows.
            self.miss_streak = self.miss_streak + 1
        else:
            target = self.target_of(agent, action)
            # A fired shot resets the streak whether or not it hits.
            self.miss_streak = 0
            roll = self.rng.random()
            hit = roll < self.accuracies[position]
            if hit:
                victim = target

        if victim is not None:
            # The killer's bonus and the victim's penalty land now, before the circle
            # contracts.
            self.add_reward(agent, self.reward_params["kill"])
            victim_agent = self.agent_of_position[victim]
            self.add_reward(victim_agent, self.reward_params["eliminated"])
            self.terminations[victim_agent] = True
            self.order.remove(victim)
            self.elimination_order.append(victim)

        # Everyone still standing gets a small nudge, under the shaped profile only.
        reward_for_surviving = self.reward_params["survive"]
        if reward_for_surviving != 0.0:
            for position_alive in self.order:
                self.add_reward(self.agent_of_position[position_alive],
                                reward_for_surviving)

        self.check_if_finished()

        for name in self.agents:
            if name == agent:
                hit_for_this_agent = hit
                action_for_this_agent = ACTION_NAMES[action]
            else:
                hit_for_this_agent = None
                action_for_this_agent = None
            self.infos[name] = {
                "t": self.t,
                "n_alive": self.n_alive(),
                "miss_streak": self.miss_streak,
                "stagnation_threshold": self.stagnation_threshold(),
                "hit": hit_for_this_agent,
                "action": action_for_this_agent,
                "termination_reason": self.reason,
            }

        if not self.episode_over:
            self.agent_selection = self.draw_actor()

        self._accumulate_rewards()
        # Put freshly dead agents at the front of the queue so they get retired.
        self._deads_step_first()

    def check_if_finished(self):
        """Three ways an episode can end."""
        if self.n_alive() == 1:
            self.reason = "sole_survivor"
            winner = self.agent_of_position[self.order[0]]
            self.add_reward(winner, self.reward_params["win"])
            self.terminations[winner] = True

        elif self.miss_streak >= self.stagnation_threshold():
            self.reason = "stagnation"
            self.split_among_survivors()
            for position in self.order:
                self.terminations[self.agent_of_position[position]] = True

        elif self.t >= self.max_steps:
            # Same payoff as stagnation. Flagged separately because a game that keeps
            # hitting this limit is a bug signal, not a result.
            self.reason = "hard_cap"
            self.split_among_survivors()
            for position in self.order:
                self.truncations[self.agent_of_position[position]] = True

        else:
            return

        self.episode_over = True

    def split_among_survivors(self):
        """Nobody won. Split the win between everyone left, so mutual abstention pays
        the same in total as a win but less each.
        """
        share = 1.0 / self.n_alive()
        for position in self.order:
            self.add_reward(self.agent_of_position[position], share)

    # ---- rendering ----

    def render(self):
        parts = []
        for position in self.order:
            name = self.agent_of_position[position]
            accuracy = self.accuracies[position]
            parts.append(name + "(p=" + format(accuracy, ".3f") + ")")
        ring = "  ".join(parts)
        return (
            "t=" + str(self.t) + "/" + str(self.max_steps)
            + "  n_alive=" + str(self.n_alive()) + "/" + str(self.n)
            + "  miss_streak=" + str(self.miss_streak)
            + "/" + str(self.stagnation_threshold()) + "\n"
            + "circle: " + ring + "  -> (wraps to first)\n"
            + "actor:  " + str(self.agent_selection)
        )

    def close(self):
        pass
