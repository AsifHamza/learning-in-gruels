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
# # Step 8 — Does the centralised critic matter past the answer key?
#
# **Deliverable 03's step 10 found** that a critic given only the actor's four numbers
# (independent PPO) passes the same gates as the centralised critic (MAPPO) at $N=3$ and
# $N=5$, and that at $N=7$, where both fail, the centralised critic is measurably better. The
# open question was whether that advantage holds, grows or disappears at 10 and 15 players.
#
# **There is no exact answer here, so there is no gate.** The comparison uses the measures this
# folder already uses where nothing can be checked:
#
# 1. **The switch test (step 2).** For every seat, does switching alone to a simple rule raise
#    its chance of winning? A profitable switch proves the policy is not an equilibrium, so
#    *fewer and smaller* profitable switches is the closest thing to "better" available here.
# 2. **Survival of the weakest**, as a multiple of an even share (step 1).
# 3. **Targeting** by rank and by players left, read against `next_weaker`, the fixed rule
#    step 3 found the agent sits on at these sizes.
# 4. **The critic's explained variance**, as in deliverable 03 step 10.
#
# **This notebook trains nothing.** `train_and_measure.py` in this folder trains and measures
# all twenty runs, in parallel, and writes one result file per run into `results/`. Training
# runs for over an hour, so it is a script; the analysis is a notebook.

# %%
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from gruel_env import GruelEnv, MISS, SHOOT_LEFT, SHOOT_RIGHT
import train_and_measure as script

PLAYER_COUNTS = script.PLAYER_COUNTS
ARMS = script.ARMS
SEEDS = script.SEEDS
SEED = 0

print("player counts:", PLAYER_COUNTS, " arms:", ARMS, " seeds:", SEEDS)
print("budget per run:", script.UPDATES, "updates x", script.EPISODES_PER_UPDATE, "episodes")
print("measured episodes per run:", script.EVALUATION_EPISODES)
print("switch test: baseline", script.BASELINE_EPISODES, "episodes, each switch",
      script.SWITCH_EPISODES, "episodes, threshold", script.STANDARD_ERRORS, "standard errors")

# %% [markdown]
# ## The result files
#
# One per run. A missing file stops the notebook, rather than letting it analyse a partial set.

# %%
R = {}
missing = []
for n in PLAYER_COUNTS:
    for arm in ARMS:
        for seed in SEEDS:
            path = script.result_path(n, arm, seed)
            if not os.path.exists(path):
                missing.append(os.path.basename(path))
                continue
            with open(path) as handle:
                R[(n, arm, seed)] = json.load(handle)

if missing:
    raise FileNotFoundError("run train_and_measure.py first; missing: " + ", ".join(missing))

print("result files read:", len(R))
for n in PLAYER_COUNTS:
    for arm in ARMS:
        minutes = [R[(n, arm, seed)]["total_minutes"] for seed in SEEDS]
        print("  N=%-3d %-11s minutes per run %s" % (n, arm, "  ".join("%.1f" % m for m in minutes)))
versions = set()
for key in R:
    versions.add((R[key]["torch"], R[key]["python"]))
print("torch and python versions across runs:", versions)

# %% [markdown]
# ## Check: the centralised arm is step 1
#
# Folder 04 step 1 trained the centralised agent at 10 and 15 players on seeds 0 to 2 with the
# same seeds and measurement. Its per-seed survival by rank is quoted here from its output. If the
# centralised arm does not reproduce it, the ablation changed something besides the critic.

# %%
STEP1_SURVIVAL = {
    (10, 0): [0.1555, 0.1368, 0.1142, 0.1069, 0.0878, 0.0791, 0.0705, 0.0684, 0.0712, 0.1096],
    (10, 1): [0.1445, 0.1306, 0.1054, 0.1018, 0.0931, 0.0833, 0.0758, 0.0706, 0.0769, 0.1180],
    (10, 2): [0.1510, 0.1252, 0.1138, 0.0956, 0.0904, 0.0804, 0.0758, 0.0721, 0.0741, 0.1216],
    (15, 0): [0.0782, 0.0683, 0.0622, 0.0644, 0.0588, 0.0539, 0.0537, 0.0504, 0.0509, 0.0543,
              0.0562, 0.0544, 0.0589, 0.0804, 0.1550],
    (15, 1): [0.1154, 0.1073, 0.0930, 0.0807, 0.0726, 0.0624, 0.0599, 0.0525, 0.0457, 0.0497,
              0.0433, 0.0398, 0.0400, 0.0523, 0.0854],
    (15, 2): [0.1097, 0.1007, 0.0977, 0.0869, 0.0826, 0.0773, 0.0743, 0.0683, 0.0605, 0.0452,
              0.0445, 0.0345, 0.0314, 0.0314, 0.0550],
}

reproduces_step1 = True
for (n, seed), expected in STEP1_SURVIVAL.items():
    here = []
    for value in R[(n, "centralised", seed)]["survival_by_rank"]:
        here.append(round(value, 4))
    same = here == expected
    if not same:
        reproduces_step1 = False
    print("  N=%-3d seed %d  %s" % (n, seed, "same as step 1" if same else "DIFFERENT"))
print()
print("centralised arm reproduces step 1:", reproduces_step1)

# %% [markdown]
# ## Classified first
#
# Stagnation and the hard cap should be zero; abstention is shown because step 1 found one seed
# of three holding fire at 15 players, and whether the critic changes that is part of the answer.

# %%
for n in PLAYER_COUNTS:
    for arm in ARMS:
        stagnation = [R[(n, arm, seed)]["stagnation_rate"] for seed in SEEDS]
        capped = [R[(n, arm, seed)]["hard_cap_rate"] for seed in SEEDS]
        abstention = [R[(n, arm, seed)]["abstention_rate"] for seed in SEEDS]
        print("  N=%-3d %-11s stagnation %s   hard cap %s"
              % (n, arm, " ".join("%.4f" % v for v in stagnation), " ".join("%.4f" % v for v in capped)))
        print("  %-17s abstention %s" % ("", " ".join("%.4f" % v for v in abstention)))
    print()

HOLDS_FIRE_THRESHOLD = 0.01
for n in PLAYER_COUNTS:
    for arm in ARMS:
        holding = 0
        for seed in SEEDS:
            if R[(n, arm, seed)]["abstention_rate"] > HOLDS_FIRE_THRESHOLD:
                holding = holding + 1
        print("  N=%-3d %-11s seeds holding fire (abstention above %.2f): %d of %d"
              % (n, arm, HOLDS_FIRE_THRESHOLD, holding, len(SEEDS)))

# %% [markdown]
# ## Which way round the circle each run fires
#
# Step 3 found that at 10 and 15 players the agent sits on `next_weaker`, which under ranked
# seating always shoots left. A policy that always shoots **right** is its mirror image: every
# player then fires at the next stronger one, and the strongest at the weakest. With $k$ players
# left, shooting left hits the stronger neighbour at 2 of the $k$ seats and shooting right at
# $k-2$ of them. So the share of shots at the stronger neighbour with the **whole field alive**
# says which way a run fires: near $2/N$ for left, near $(N-2)/N$ for right. A run is called
# `left` below 0.5, `right` above it, and the share is printed so a run between the two is visible.

# %%
MODE = {}
print("  full field: shooting left gives 2/N, shooting right gives (N-2)/N")
for n in PLAYER_COUNTS:
    print("  N=%d   2/N = %.4f   (N-2)/N = %.4f" % (n, 2.0 / n, (n - 2.0) / n))
    for arm in ARMS:
        cells = []
        for seed in SEEDS:
            share = R[(n, arm, seed)]["targeting_by_alive"][str(n)]
            mode = "left" if share < 0.5 else "right"
            MODE[(n, arm, seed)] = mode
            cells.append("%.4f %-5s" % (share, mode))
        print("    %-11s %s" % (arm, "   ".join(cells)))
    print()

for n in PLAYER_COUNTS:
    for arm in ARMS:
        lefts = 0
        for seed in SEEDS:
            if MODE[(n, arm, seed)] == "left":
                lefts = lefts + 1
        print("  N=%-3d %-11s fires left in %d of %d seeds, right in %d" % (n, arm, lefts, len(SEEDS), len(SEEDS) - lefts))

# %% [markdown]
# **Both directions are learned under both critics.** Which one a run lands on is a property of the
# seed, not of what the critic sees, and the two directions are different policies with different
# survival profiles. That matters for every comparison below: a seed where one arm fired left and
# the other right contributes a large difference that says nothing about the critic. So each
# comparison is also given restricted to the seeds where **both** arms fired left.

# %%
BOTH_LEFT = {}
for n in PLAYER_COUNTS:
    BOTH_LEFT[n] = []
    for seed in SEEDS:
        if MODE[(n, "centralised", seed)] == "left" and MODE[(n, "local", seed)] == "left":
            BOTH_LEFT[n].append(seed)
    print("  N=%-3d seeds where both arms fire left: %s" % (n, BOTH_LEFT[n]))

# %% [markdown]
# **Where the two arms play identically.** If both actors converge to the same effectively
# deterministic policy, the shared seeds make every sampled game the same, and the two runs then
# measure identically however different their critics are. Checked per seed.

# %%
for n in PLAYER_COUNTS:
    for seed in SEEDS:
        central = R[(n, "centralised", seed)]
        local = R[(n, "local", seed)]
        same_play = (central["survival_by_rank"] == local["survival_by_rank"]
                     and central["switch_test"] == local["switch_test"])
        if same_play:
            print("  N=%-3d seed %d: both arms measure identically (survival, targeting, every switch);"
                  % (n, seed))
            print("      targeting distance from 2/k at the full field %.4f; critic EV %.4f against %.4f"
                  % (abs(central["targeting_by_alive"][str(n)] - 2.0 / n),
                     central["explained_variance"], local["explained_variance"]))

# %% [markdown]
# ## The paired comparison, and the rule for reading it
#
# Each seed gives one centralised and one local run that start from the same actor. Every measure
# is compared seed by seed: the mean of the five differences, centralised minus local, with the
# standard error of that mean. **Fixed before the results were read:** a difference counts only if
# it exceeds twice its standard error. With five seeds that is a loose bar, so a difference that
# only just clears it is reported as such, and a subset of fewer than three seeds gets no verdict.
# Only two measures have a direction that means
# "better" without an answer key: fewer profitable switches and lower exploitability, and a
# higher explained variance for the critic. The others report "higher" or "lower".

# %%
def mean_of(values):
    return sum(values) / len(values)


def standard_error_of_mean(values):
    centre = mean_of(values)
    total = 0.0
    for value in values:
        total = total + (value - centre) ** 2
    return (total / (len(values) - 1) / len(values)) ** 0.5


def paired(measure, n, higher_is_better, seeds=None):
    if seeds is None:
        seeds = SEEDS
    central = [measure(R[(n, "centralised", seed)]) for seed in seeds]
    local = [measure(R[(n, "local", seed)]) for seed in seeds]
    differences = []
    for index in range(len(seeds)):
        differences.append(central[index] - local[index])
    gap = mean_of(differences)
    if len(seeds) < 3:
        # One or two differences give no usable estimate of their own spread.
        return central, local, gap, float("nan"), "too few seeds to judge"
    error = standard_error_of_mean(differences)
    if abs(gap) <= 2 * error:
        verdict = "no measurable difference"
    elif higher_is_better is None:
        # No answer key says which direction is better, so only the direction is reported.
        verdict = "centralised higher" if gap > 0 else "centralised lower"
    elif (gap > 0) == higher_is_better:
        verdict = "centralised better"
    else:
        verdict = "local better"
    return central, local, gap, error, verdict


def show(label, n, result):
    central, local, gap, error, verdict = result
    print("  N=%-3d %s" % (n, label))
    if len(central) != len(SEEDS):
        print("      (seeds where both arms fire left only)")
    print("      centralised %s   mean %.4f" % (" ".join("%.4f" % v for v in central), mean_of(central)))
    print("      local       %s   mean %.4f" % (" ".join("%.4f" % v for v in local), mean_of(local)))
    print("      difference %+.4f, s.e. %.4f  -> %s" % (gap, error, verdict))

# %% [markdown]
# ## 1. The switch test
#
# Two numbers per run. The **count** of profitable switches out of $4N$ tried. And the
# **exploitability**: for each seat, the largest gain any of the four switches gave it, averaged
# over the seats. A policy that is an equilibrium against these four rules has a count of zero
# and an exploitability at or below zero.

# %%
def profitable_count(result):
    count = 0
    for row in result["switch_test"]:
        if row["profitable"]:
            count = count + 1
    return count


def exploitability(result):
    best = {}
    for row in result["switch_test"]:
        if row["seat"] not in best or row["gain"] > best[row["seat"]]:
            best[row["seat"]] = row["gain"]
    return mean_of(list(best.values()))


SWITCH_COUNT = {}
EXPLOIT = {}
for n in PLAYER_COUNTS:
    SWITCH_COUNT[n] = paired(profitable_count, n, higher_is_better=False)
    show("profitable switches, of %d tried per run" % (4 * n), n, SWITCH_COUNT[n])
    EXPLOIT[n] = paired(exploitability, n, higher_is_better=False)
    show("exploitability: best gain from switching, mean over seats", n, EXPLOIT[n])
    show("profitable switches", n, paired(profitable_count, n, False, BOTH_LEFT[n]))
    show("exploitability", n, paired(exploitability, n, False, BOTH_LEFT[n]))
    print()

# %% [markdown]
# **Who can gain, and by switching to what.** Step 2 found every profitable switch at 15 players
# was made by a strong player short of the strongest, mostly to `shoot_strongest`. Listed for both
# arms, so a change in *where* the policy is weak is visible as well as a change in how often.

# %%
for n in PLAYER_COUNTS:
    for arm in ARMS:
        by_seat = {}
        by_rule = {}
        for seed in SEEDS:
            for row in R[(n, arm, seed)]["switch_test"]:
                if row["profitable"]:
                    by_seat[row["seat"]] = by_seat.get(row["seat"], 0) + 1
                    by_rule[row["switch"]] = by_rule.get(row["switch"], 0) + 1
        seats = "  ".join("seat %d: %d" % (seat, by_seat[seat]) for seat in sorted(by_seat))
        rules = "  ".join("%s: %d" % (rule, by_rule[rule]) for rule in sorted(by_rule))
        print("  N=%-3d %-11s by seat (0 = weakest)  %s" % (n, arm, seats if seats else "none"))
        print("  %-17s by rule               %s" % ("", rules if rules else "none"))
    print()

# %% [markdown]
# **The switch test by direction**, pooling both arms: whether one direction leaves more for a
# single player to gain by switching.

# %%
for n in PLAYER_COUNTS:
    for mode in ["left", "right"]:
        counts = []
        exploits = []
        for arm in ARMS:
            for seed in SEEDS:
                if MODE[(n, arm, seed)] == mode:
                    counts.append(profitable_count(R[(n, arm, seed)]))
                    exploits.append(exploitability(R[(n, arm, seed)]))
        if counts:
            print("  N=%-3d %-5s runs %d   profitable switches %s   mean exploitability %+.4f"
                  % (n, mode, len(counts), " ".join(str(c) for c in counts), mean_of(exploits)))
        else:
            print("  N=%-3d %-5s runs 0" % (n, mode))

# %% [markdown]
# ## 2. Survival of the weakest

# %%
def weakest_advantage(result):
    return result["survival_by_rank"][0] * result["n"]


ADVANTAGE = {}
for n in PLAYER_COUNTS:
    ADVANTAGE[n] = paired(weakest_advantage, n, higher_is_better=None)
    show("weakest player's survival as a multiple of 1/N", n, ADVANTAGE[n])
    show("weakest player's survival as a multiple of 1/N", n,
         paired(weakest_advantage, n, None, BOTH_LEFT[n]))
    print()

print("  survival by rank, pooled over seeds (rank 0 = weakest)")
for n in PLAYER_COUNTS:
    for arm in ARMS:
        pooled = []
        for rank in range(n):
            pooled.append(mean_of([R[(n, arm, seed)]["survival_by_rank"][rank] for seed in SEEDS]))
        print("  N=%-3d %-11s %s" % (n, arm, " ".join("%.4f" % v for v in pooled)))

# %% [markdown]
# **A higher weakest-player advantage is not "better" on its own**, so this comparison, like the
# targeting ones below, reports only a direction. With no exact answer it is
# not known what the equilibrium's value is at these sizes. It is reported because it is the
# headline quantity of the project, and the switch test above is what bears on equilibrium.

# %% [markdown]
# ## 3. Targeting, against `next_weaker`
#
# `next_weaker` is Archetti's rule restricted to the two neighbours, which under ranked seating
# always shoots left (step 3). It is measured here in the same game, over 10,000 episodes.

# %%
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


REFERENCE = {}
for n in PLAYER_COUNTS:
    game = GruelEnv(n=n, reward=script.REWARD, seating=script.SEATING, seed=7000 + n)
    records = script.play_episodes(game, next_weaker, 10000)
    REFERENCE[n] = script.summarise(records, n)
    print("  N=%-3d next_weaker by rank %s" % (n, " ".join("%.4f" % v for v in REFERENCE[n]["targeting_by_rank"])))


def distance(first, second):
    total = 0.0
    for a, b in zip(first, second):
        total = total + abs(a - b)
    return total / len(first)

# %%
print()
for n in PLAYER_COUNTS:
    for arm in ARMS:
        pooled = []
        for rank in range(n):
            pooled.append(mean_of([R[(n, arm, seed)]["targeting_by_rank"][rank] for seed in SEEDS]))
        print("  N=%-3d %-11s by rank %s" % (n, arm, " ".join("%.4f" % v for v in pooled)))
    print()

DISTANCE_TO_RULE = {}
for n in PLAYER_COUNTS:
    def from_rule(result, n=n):
        return distance(result["targeting_by_rank"], REFERENCE[n]["targeting_by_rank"])
    DISTANCE_TO_RULE[n] = paired(from_rule, n, higher_is_better=None)
    show("targeting distance from next_weaker, by rank", n, DISTANCE_TO_RULE[n])
    show("targeting distance from next_weaker, by rank", n,
         paired(from_rule, n, None, BOTH_LEFT[n]))
    print()

for n in PLAYER_COUNTS:
    between = []
    for seed in SEEDS:
        between.append(distance(R[(n, "centralised", seed)]["targeting_by_rank"],
                                R[(n, "local", seed)]["targeting_by_rank"]))
    print("  N=%-3d distance between the two arms, per seed %s   mean %.4f"
          % (n, " ".join("%.4f" % v for v in between), mean_of(between)))

# %% [markdown]
# **With three players left** is where step 4 found the agent departs from `next_weaker`, and so
# where a difference in what the critic sees could show in behaviour.

# %%
for n in PLAYER_COUNTS:
    rule_three = REFERENCE[n]["targeting_by_alive"]["3"]
    def three_left(result):
        return result["targeting_by_alive"]["3"]
    THREE = paired(three_left, n, higher_is_better=None)
    show("P(shoot the stronger neighbour) with three left; next_weaker gives %.4f" % rule_three, n, THREE)
    print()

# %% [markdown]
# ## 4. The critic's explained variance
#
# $\text{EV} = 1 - \operatorname{Var}(G - V)/\operatorname{Var}(G)$ on a fresh 200-episode batch,
# where $G$ is the return a decision went on to receive and $V$ the critic's prediction.

# %%
EV = {}
for n in PLAYER_COUNTS:
    def explained(result):
        return result["explained_variance"]
    EV[n] = paired(explained, n, higher_is_better=True)
    show("explained variance", n, EV[n])
    print()

# %% [markdown]
# ## The picture

# %%
figure, axes = plt.subplots(1, 2, figsize=(10, 4))
offsets = {"centralised": -0.08, "local": 0.08}
markers = {"centralised": "o", "local": "s"}
colours = {"centralised": "tab:blue", "local": "tab:red"}
panels = [("profitable switches per run", profitable_count),
          ("exploitability (mean best gain per seat)", exploitability)]
for axis, (title, measure) in zip(axes, panels):
    for arm in ARMS:
        for n_index in range(len(PLAYER_COUNTS)):
            n = PLAYER_COUNTS[n_index]
            values = [measure(R[(n, arm, seed)]) for seed in SEEDS]
            label = arm if n_index == 0 else None
            axis.scatter([n_index + offsets[arm]] * len(values), values, marker=markers[arm],
                         color=colours[arm], label=label, alpha=0.8)
    axis.set_xticks(range(len(PLAYER_COUNTS)))
    axis.set_xticklabels(["N=%d" % n for n in PLAYER_COUNTS])
    axis.set_title(title)
    axis.grid(True, axis="y")
axes[1].axhline(0.0, color="black", linewidth=0.8)
axes[0].legend()
figure.suptitle("Switch test, centralised critic against local critic, five seeds each")
figure.tight_layout()
plt.show()

# %% [markdown]
# ## What this step establishes, derived from the measurements

# %%
print("centralised arm reproduces step 1:", reproduces_step1)
print()
for n in PLAYER_COUNTS:
    for arm in ARMS:
        lefts = sum(1 for seed in SEEDS if MODE[(n, arm, seed)] == "left")
        print("  N=%-3d %-11s fires left in %d of %d seeds" % (n, arm, lefts, len(SEEDS)))
print()
summary_rows = [("profitable switches", SWITCH_COUNT), ("exploitability", EXPLOIT),
                ("weakest advantage", ADVANTAGE), ("distance from next_weaker", DISTANCE_TO_RULE),
                ("explained variance", EV)]
for n in PLAYER_COUNTS:
    print("  N=%d" % n)
    for label, table in summary_rows:
        central, local, gap, error, verdict = table[n]
        print("    %-28s centralised %.4f  local %.4f  difference %+.4f (s.e. %.4f)  %s"
              % (label, mean_of(central), mean_of(local), gap, error, verdict))
    print()

equilibrium_verdicts = []
for n in PLAYER_COUNTS:
    equilibrium_verdicts.append((n, SWITCH_COUNT[n][4], EXPLOIT[n][4]))
helped = [n for n, count_verdict, exploit_verdict in equilibrium_verdicts
          if "centralised better" in (count_verdict, exploit_verdict)]
hurt = [n for n, count_verdict, exploit_verdict in equilibrium_verdicts
        if "local better" in (count_verdict, exploit_verdict)]
if helped:
    print("On the switch test the centralised critic is measurably closer to an equilibrium at N =", helped)
if hurt:
    print("On the switch test the local critic is measurably closer to an equilibrium at N =", hurt)
if not helped and not hurt:
    print("On the switch test neither critic is measurably closer to an equilibrium at 10 or 15 players.")

# %% [markdown]
# ## Reproducibility
#
# Every job in the script seeds itself, so one job run twice must give an identical result file.
# Checked here on a shortened job (five updates, small measurements), which exercises the same
# code path in seconds.

# %%
def shortened_job():
    saved = (script.UPDATES, script.EVALUATION_EPISODES, script.BASELINE_EPISODES,
             script.SWITCH_EPISODES, script.CRITIC_EPISODES)
    script.UPDATES = 5
    script.EVALUATION_EPISODES = 200
    script.BASELINE_EPISODES = 200
    script.SWITCH_EPISODES = 100
    script.CRITIC_EPISODES = 20
    try:
        model = script.train_one(10, "local", 0)
        game = GruelEnv(n=10, reward=script.REWARD, seating=script.SEATING, seed=5000)
        torch.manual_seed(7000)
        summary = script.summarise(script.play_episodes(game, script.make_agent_policy(model), 200), 10)
        summary["switch_test"] = script.switch_test(10, model, 0)
    finally:
        (script.UPDATES, script.EVALUATION_EPISODES, script.BASELINE_EPISODES,
         script.SWITCH_EPISODES, script.CRITIC_EPISODES) = saved
    return json.dumps(summary, sort_keys=True)


first = shortened_job()
second = shortened_job()
print("identical:", first == second)
