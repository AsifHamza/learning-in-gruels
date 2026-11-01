"""Watch a gruel episode in a live window.

This is a script and not a notebook because it opens a window, and a window cannot
live inside a notebook.

    ./.venv/bin/python play.py --n 6
    ./.venv/bin/python play.py --n 15 --policy random --fps 6
    ./.venv/bin/python play.py --n 5 --step

With --step: SPACE advances one frame, P plays on without pausing, Q quits.
"""

import argparse
import sys

import numpy as np
import pygame

SHOOT_LEFT = 0
SHOOT_RIGHT = 1
MISS = 2

# Colours, as (red, green, blue).
BACKGROUND = (250, 250, 252)
EDGE_LIVE = (203, 205, 211)
EDGE_CONTRACTED = (232, 140, 40)
SHOT_HIT = (206, 60, 60)
SHOT_MISSED = (86, 88, 96)
NODE_FILL = (255, 255, 255)
NODE_EDGE = (40, 42, 50)
NODE_DEAD = (196, 198, 205)
NODE_ACTING = (40, 110, 210)
TEXT = (30, 32, 40)
TEXT_DEAD = (150, 152, 160)


class Gruel:
    """The same game as the notebooks."""

    def __init__(self, n=3, seating="random", seed=0):
        self.n = n
        self.seating = seating
        self.rng = np.random.default_rng(seed)
        self.stagnation_rounds = 10
        self.max_steps = 200 * n

    def reset(self):
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
        self.finished = False
        self.reason = None
        self.rewards = np.zeros(self.n)
        self.actor = int(self.rng.choice(self.order))

    def n_alive(self):
        return len(self.order)

    def neighbours(self, position):
        index_in_order = self.order.index(position)
        how_many_alive = len(self.order)
        left = self.order[(index_in_order - 1) % how_many_alive]
        right = self.order[(index_in_order + 1) % how_many_alive]
        return left, right

    def observe(self, position):
        left, right = self.neighbours(position)
        return np.array([
            self.accuracies[position],
            self.accuracies[left],
            self.accuracies[right],
            self.n_alive() / self.n,
        ])

    def live_edges(self):
        edges = []
        how_many_alive = len(self.order)
        if how_many_alive < 2:
            return edges
        for index_in_order in range(how_many_alive):
            this_player = self.order[index_in_order]
            next_player = self.order[(index_in_order + 1) % how_many_alive]
            edges.append((this_player, next_player))
            if how_many_alive == 2:
                break
        return edges

    def target_of(self, position, action):
        if action == MISS:
            return None
        left, right = self.neighbours(position)
        if action == SHOOT_LEFT:
            return left
        else:
            return right

    def step(self, action):
        shooter = self.actor
        self.t = self.t + 1

        target = None
        victim = None
        hit = False

        if action == MISS:
            self.miss_streak = self.miss_streak + 1
        else:
            target = self.target_of(shooter, action)
            self.miss_streak = 0
            roll = self.rng.random()
            hit = roll < self.accuracies[shooter]
            if hit:
                victim = target

        if victim is not None:
            self.order.remove(victim)

        if self.n_alive() == 1:
            self.finished = True
            self.reason = "sole_survivor"
            self.rewards[self.order[0]] = 1.0
        elif self.miss_streak >= self.stagnation_rounds * self.n_alive():
            self.finished = True
            self.reason = "stagnation"
            share = 1.0 / self.n_alive()
            for position in self.order:
                self.rewards[position] = share
        elif self.t >= self.max_steps:
            self.finished = True
            self.reason = "hard_cap"

        if not self.finished:
            self.actor = int(self.rng.choice(self.order))

        return {"shooter": shooter, "action": action, "target": target,
                "hit": hit, "victim": victim}


# ---- policies ----

def shoot_strongest(observation, rng):
    if observation[1] > observation[2]:
        return SHOOT_LEFT
    else:
        return SHOOT_RIGHT


def shoot_weakest(observation, rng):
    if observation[1] < observation[2]:
        return SHOOT_LEFT
    else:
        return SHOOT_RIGHT


def random_policy(observation, rng):
    return int(rng.integers(3))


def always_miss(observation, rng):
    return MISS


POLICIES = {
    "shoot_strongest": shoot_strongest,
    "shoot_weakest": shoot_weakest,
    "random": random_policy,
    "always_miss": always_miss,
}


# ---- drawing ----

def node_screen_positions(n, centre_x, centre_y, radius):
    """Fixed screen pixel for every position. Position 0 at the top, going clockwise.

    Computed once from the original n. Survivors are never re-spaced, so the gap a dead
    player leaves stays visible and the contracted edge visibly jumps across it.
    """
    coordinates = []
    for i in range(n):
        angle = np.pi / 2 - 2 * np.pi * i / n
        x = centre_x + radius * np.cos(angle)
        # Minus, because screen y grows downwards while maths y grows upwards.
        y = centre_y - radius * np.sin(angle)
        coordinates.append((int(x), int(y)))
    return coordinates


def draw_dashed_line(surface, colour, start, end, width=3, dash=9, gap=7):
    """pygame has no dashed line, so draw a row of short segments."""
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    length = float(np.hypot(dx, dy))
    if length < 1:
        return

    step_x = dx / length
    step_y = dy / length

    travelled = 0.0
    while travelled < length:
        segment_end = min(travelled + dash, length)
        start_point = (x1 + step_x * travelled, y1 + step_y * travelled)
        end_point = (x1 + step_x * segment_end, y1 + step_y * segment_end)
        pygame.draw.line(surface, colour, start_point, end_point, width)
        travelled = segment_end + gap


def draw_arrow(surface, colour, start, end, width=4):
    """A line with a triangular head, pulled back so it does not cover the nodes."""
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    length = float(np.hypot(dx, dy))
    if length < 1:
        return

    step_x = dx / length
    step_y = dy / length

    pull_back = 34
    from_x = x1 + step_x * pull_back
    from_y = y1 + step_y * pull_back
    to_x = x2 - step_x * pull_back
    to_y = y2 - step_y * pull_back

    pygame.draw.line(surface, colour, (from_x, from_y), (to_x, to_y), width)

    head = 13
    left_x = to_x - step_x * head - step_y * head * 0.55
    left_y = to_y - step_y * head + step_x * head * 0.55
    right_x = to_x - step_x * head + step_y * head * 0.55
    right_y = to_y - step_y * head - step_x * head * 0.55
    pygame.draw.polygon(surface, colour,
                        [(to_x, to_y), (left_x, left_y), (right_x, right_y)])


def caption_for(game, event):
    """One line of plain English describing what just happened."""
    if event is None:
        return f"{game.n_alive()} players, nobody has acted yet"

    shooter = event["shooter"]
    shooter_accuracy = game.accuracies[shooter]

    if event["action"] == MISS:
        return f"p{shooter} ({shooter_accuracy:.0%}) abstains - a deliberate miss"

    target = event["target"]
    target_accuracy = game.accuracies[target]
    start = f"p{shooter} ({shooter_accuracy:.0%}) shoots p{target} ({target_accuracy:.0%})"

    if event["hit"]:
        return start + f" - HIT, p{target} is out"
    else:
        return start + " - missed"


def draw(surface, fonts, game, layout, original_edges, event, message):
    surface.fill(BACKGROUND)
    font_small, font_medium, font_large = fonts

    # Edges.
    for edge in game.live_edges():
        first, second = edge
        was_original = (first, second) in original_edges or (second, first) in original_edges
        if was_original:
            pygame.draw.line(surface, EDGE_LIVE, layout[first], layout[second], 3)
        else:
            draw_dashed_line(surface, EDGE_CONTRACTED, layout[first], layout[second], 3)

    # The shot.
    if event is not None and event["target"] is not None:
        if event["hit"]:
            colour = SHOT_HIT
        else:
            colour = SHOT_MISSED
        draw_arrow(surface, colour, layout[event["shooter"]], layout[event["target"]])

    # Players.
    for position in range(game.n):
        x, y = layout[position]
        alive = position in game.order

        if alive:
            outline = NODE_EDGE
            if position == game.actor and not game.finished:
                outline = NODE_ACTING
            pygame.draw.circle(surface, NODE_FILL, (x, y), 26)
            pygame.draw.circle(surface, outline, (x, y), 26, 3)
            label_colour = TEXT
        else:
            pygame.draw.circle(surface, (243, 243, 246), (x, y), 26)
            pygame.draw.circle(surface, NODE_DEAD, (x, y), 26, 2)
            pygame.draw.line(surface, SHOT_HIT, (x - 17, y), (x + 17, y), 3)
            label_colour = TEXT_DEAD

        name = font_small.render(f"p{position}", True, label_colour)
        accuracy = font_small.render(f"{game.accuracies[position]:.0%}", True, label_colour)
        surface.blit(name, (x - name.get_width() // 2, y - 15))
        surface.blit(accuracy, (x - accuracy.get_width() // 2, y + 1))

    # Caption and status.
    caption = font_large.render(message, True, TEXT)
    surface.blit(caption, (24, 18))

    status = (f"t={game.t}   alive={game.n_alive()}/{game.n}   "
              f"deliberate misses in a row={game.miss_streak}"
              f"/{game.stagnation_rounds * game.n_alive()}")
    surface.blit(font_medium.render(status, True, (110, 112, 120)), (24, 46))

    # Legend, bottom left, saying what each mark means.
    legend_y = surface.get_height() - 92
    pygame.draw.line(surface, EDGE_LIVE, (28, legend_y), (74, legend_y), 3)
    surface.blit(font_small.render("these two started as neighbours", True, TEXT),
                 (84, legend_y - 7))

    draw_dashed_line(surface, EDGE_CONTRACTED, (28, legend_y + 22), (74, legend_y + 22), 3)
    surface.blit(font_small.render("became neighbours because someone died", True, TEXT),
                 (84, legend_y + 15))

    pygame.draw.line(surface, SHOT_HIT, (28, legend_y + 44), (74, legend_y + 44), 3)
    surface.blit(font_small.render("a shot that hit", True, TEXT), (84, legend_y + 37))

    pygame.draw.line(surface, SHOT_MISSED, (28, legend_y + 66), (74, legend_y + 66), 3)
    surface.blit(font_small.render("a shot that missed", True, TEXT), (84, legend_y + 59))

    if game.finished:
        ending = f"ended: {game.reason}"
        surface.blit(font_large.render(ending, True, SHOT_HIT),
                     (surface.get_width() - 240, 18))


def wait_for_keypress():
    """Returns 'next', 'play' or 'quit'."""
    while True:
        for pygame_event in pygame.event.get():
            if pygame_event.type == pygame.QUIT:
                return "quit"
            if pygame_event.type == pygame.KEYDOWN:
                if pygame_event.key in (pygame.K_SPACE, pygame.K_RIGHT, pygame.K_RETURN):
                    return "next"
                if pygame_event.key == pygame.K_p:
                    return "play"
                if pygame_event.key in (pygame.K_q, pygame.K_ESCAPE):
                    return "quit"
        pygame.time.wait(30)


def main():
    parser = argparse.ArgumentParser(description="Watch a gruel episode.")
    parser.add_argument("--n", type=int, default=6, help="number of players")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", default="shoot_strongest", choices=sorted(POLICIES))
    parser.add_argument("--seating", default="random", choices=["random", "ranked"])
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--step", action="store_true",
                        help="pause on every frame; SPACE advances, P plays on, Q quits")
    args = parser.parse_args()

    if args.n < 2:
        print("need at least 2 players")
        return 1

    pygame.init()
    width = 1000
    height = 720
    surface = pygame.display.set_mode((width, height))
    pygame.display.set_caption(f"Gruel — N={args.n}, {args.policy}")

    fonts = (
        pygame.font.SysFont("helvetica", 13),
        pygame.font.SysFont("helvetica", 15),
        pygame.font.SysFont("helvetica", 19),
    )
    clock = pygame.time.Clock()

    policy = POLICIES[args.policy]
    game = Gruel(n=args.n, seating=args.seating, seed=args.seed)

    radius = min(width, height) * 0.33
    layout = node_screen_positions(args.n, width // 2, height // 2 + 10, radius)

    stepping = args.step
    running = True

    for episode in range(args.episodes):
        if not running:
            break

        game.reset()
        original_edges = set(game.live_edges())
        event = None

        while running:
            for pygame_event in pygame.event.get():
                if pygame_event.type == pygame.QUIT:
                    running = False
                if pygame_event.type == pygame.KEYDOWN:
                    if pygame_event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False

            if not running:
                break

            draw(surface, fonts, game, layout, original_edges, event, caption_for(game, event))
            pygame.display.flip()

            if stepping:
                key = wait_for_keypress()
                if key == "quit":
                    running = False
                    break
                if key == "play":
                    stepping = False
            else:
                clock.tick(args.fps)

            if game.finished:
                break

            observation = game.observe(game.actor)
            action = policy(observation, game.rng)
            event = game.step(action)

        # Hold on the final state so you can see who won.
        if running:
            draw(surface, fonts, game, layout, original_edges, event, caption_for(game, event))
            pygame.display.flip()
            if stepping:
                if wait_for_keypress() == "quit":
                    running = False
            else:
                pygame.time.wait(1500)

        print(f"episode {episode}: ended {game.reason} after {game.t} steps, "
              f"survivors {game.order}, "
              f"winner accuracy {[round(float(game.accuracies[p]), 2) for p in game.order]}")

    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
