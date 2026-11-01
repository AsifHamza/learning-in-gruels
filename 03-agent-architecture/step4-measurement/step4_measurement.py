# %% [markdown]
# # Step 4 — The measurement machinery
#
# This step passes no gate. It is what *decides* one, which makes it the place where a
# mistake is most expensive: **a broken metric does not look like a bug, it looks like a
# result.**
#
# So every instrument here is built and then pointed at **heuristic policies whose
# behaviour is known in advance**, before any of them is ever pointed at a trained agent.
# If an instrument disagrees with a heuristic whose answer is obvious, the instrument is
# wrong.
#
# Six things get built, in the order they are applied:
#
# 1. **Classify the equilibrium** — before looking at any survival number.
# 2. **Count survival correctly** — from who is alive, never from rewards.
# 3. **Count targeting honestly** — and distinguish *no decisions* from *a rate of zero*.
# 4. **The ratio gate** — measured against an exact target. Works up to N=7.
# 5. **The focal-agent protocol** — measured against baselines. The only option above that.
# 6. **Honest error bars** — and the difference between an interval and a test.

# %%
import numpy as np

from gruel_env import GruelEnv, MISS, SHOOT_LEFT, SHOOT_RIGHT

SEED = 0
rng = np.random.default_rng(SEED)

EXACT_N3 = {0.4: 10 / 27, 0.6: 9 / 27, 0.8: 8 / 27}

print("exact N=3 targets:", {a: round(v, 5) for a, v in EXACT_N3.items()})

# %% [markdown]
# ## The policies to calibrate against
#
# Three rules whose behaviour needs no measurement to predict. That is exactly why they
# are the right things to test an instrument on.

# %%
def shoot_strongest(game, actor):
    """Always fire at whichever neighbour shoots better."""
    left, right = game.neighbours(actor)
    if game.accuracies[left] >= game.accuracies[right]:
        return SHOOT_LEFT
    return SHOOT_RIGHT


def always_miss(game, actor):
    """Never fire. Every episode should end in stagnation."""
    return MISS


def random_policy(game, actor):
    """Uniform over the three actions."""
    return int(rng.integers(0, 3))


# %% [markdown]
# ## 1. Classify first, look second
#
# This game has two kinds of equilibrium. In the **shooting** equilibrium players fire and
# someone wins. In the **non-shooting** one everyone abstains forever and the episode is
# cut off by the stagnation rule.
#
# Those are different games with different survival profiles, and the analytical targets
# describe only the first. So the stagnation rate is checked **before** any survival number
# is read — otherwise you average two populations and get a number describing neither.


# %%
def play_episodes(game, policy, episodes):
    """Play `episodes` episodes. Returns one record per episode."""
    records = []
    for _ in range(episodes):
        game.reset()
        decisions = []

        for acting_agent in game.agent_iter():
            observation, reward, termination, truncation, info = game.last()

            # A finished agent is handed to us one last time, so PettingZoo can retire
            # it. Stepping with None is how we say 'nothing to do here'.
            if termination or truncation:
                game.step(None)
                continue

            actor = game.position_of_agent[acting_agent]
            action = policy(game, actor)

            # Record the CONTEXT BEFORE stepping. A kill contracts the circle, so after
            # the step this shooter's neighbours are no longer the ones it aimed at.
            left, right = game.neighbours(actor)
            decisions.append({
                "accuracy": round(float(game.accuracies[actor]), 4),
                "action": action,
                "left_accuracy": round(float(game.accuracies[left]), 4),
                "right_accuracy": round(float(game.accuracies[right]), 4),
                "n_alive": game.n_alive(),
            })

            game.step(action)

        # Read the totals from the environment's own running tally. PettingZoo clears the
        # per-step rewards every time an agent acts, so a per-step sum would drop
        # everything except the last step.
        totals = {}
        for position in range(game.n):
            name = game.agent_of_position[position]
            totals[position] = game.episode_returns[name]

        if len(game.order) == 1:
            survivor_accuracy = round(float(game.accuracies[game.order[0]]), 4)
        else:
            survivor_accuracy = None

        records.append({
            "total_reward": dict(totals),
            "survivors": list(game.order),
            "survivor_accuracy": survivor_accuracy,
            "stagnated": game.reason == "stagnation",
            "hit_cap": game.reason == "hard_cap",
            "steps": game.t,
            "decisions": decisions,
            "accuracies": [round(float(a), 4) for a in game.accuracies],
        })
    return records


def stagnation_rate(records):
    stagnated = 0
    for record in records:
        if record["stagnated"]:
            stagnated = stagnated + 1
    return stagnated / len(records)


def cap_rate(records):
    hit = 0
    for record in records:
        if record["hit_cap"]:
            hit = hit + 1
    return hit / len(records)


# The same 0.01 cut that steps 6 and 9 apply to trained agents, so a policy is called
# "shooting" here by the rule that later decides whether a gate may be read.
SHOOTING_LIMIT = 0.01


def classify(records):
    rate = stagnation_rate(records)
    if rate < SHOOTING_LIMIT:
        return "shooting"
    if rate > 1 - SHOOTING_LIMIT:
        return "non_shooting"
    return "mixed"


game = GruelEnv(n=3, reward="terminal", seating="ranked", seed=1000 + SEED)
shooting_records = play_episodes(game, shoot_strongest, 3000)
abstain_records = play_episodes(game, always_miss, 300)
random_records = play_episodes(game, random_policy, 3000)

print("%-18s %11s %8s  %s" % ("policy", "stagnation", "cap", "class"))
for name, records in [("shoot_strongest", shooting_records),
                      ("always_miss", abstain_records),
                      ("random", random_records)]:
    print("%-18s %11.4f %8.4f  %s"
          % (name, stagnation_rate(records), cap_rate(records), classify(records)))

# %% [markdown]
# `always_miss` stagnates in **every** episode, which is the instrument agreeing with
# something obvious. If it did not, the classifier would be broken.
#
# **The cap rate is a bug signal, not a result.** It counts episodes cut off by the hard
# step limit. A shooting policy should never reach it; if it does, something is wrong with
# the environment rather than interesting about the policy.

# %% [markdown]
# ## 2. Count survival from survivors, not from rewards
#
# The tempting shortcut is to average each player's episode reward. Under the `terminal`
# profile that happens to equal survival probability, because the only payoff is +1 for
# winning.
#
# **Under `shaped` it does not**, because kills and per-step bonuses are in there too. So
# survival is always counted by asking who was alive at the end. That makes it
# reward-independent, and comparable across every run in the project.


# %%
def survival_by_accuracy(records):
    """Fraction of episodes each accuracy was the sole survivor."""
    wins = {}
    total = len(records)
    for record in records:
        for accuracy in record["accuracies"]:
            if accuracy not in wins:
                wins[accuracy] = 0
    for record in records:
        if record["survivor_accuracy"] is not None:
            wins[record["survivor_accuracy"]] = wins[record["survivor_accuracy"]] + 1

    probabilities = {}
    for accuracy in wins:
        probabilities[accuracy] = wins[accuracy] / total
    return probabilities


def mean_total_reward(records):
    """Average reward each accuracy actually collected — read from the environment's
    own running total, not recomputed from who survived."""
    totals = {}
    counts = {}
    for record in records:
        for position in range(len(record["accuracies"])):
            seat_accuracy = record["accuracies"][position]
            totals[seat_accuracy] = (totals.get(seat_accuracy, 0.0)
                                     + record["total_reward"][position])
            counts[seat_accuracy] = counts.get(seat_accuracy, 0) + 1
    means = {}
    for accuracy in totals:
        means[accuracy] = totals[accuracy] / counts[accuracy]
    return means


shaped_game = GruelEnv(n=3, reward="shaped", seating="ranked", seed=1000 + SEED)
shaped_records = play_episodes(shaped_game, shoot_strongest, 3000)

survival = survival_by_accuracy(shooting_records)
terminal_reward = mean_total_reward(shooting_records)
shaped_survival = survival_by_accuracy(shaped_records)
shaped_reward = mean_total_reward(shaped_records)

print("  %4s %10s %13s %13s" % ("acc", "survival", "mean return", "mean return"))
print("  %4s %10s %13s %13s" % ("", "(terminal)", "terminal", "shaped"))
for accuracy in sorted(survival, reverse=True):
    print("  %4s %10.4f %13.4f %13.4f"
          % (accuracy, survival[accuracy], terminal_reward[accuracy],
             shaped_reward[accuracy]))
print()
print("Under `terminal` the return column IS the survival column -- the only payoff is")
print("+1 for winning. Under `shaped` it is not: kills, the elimination penalty and")
print("per-step bonuses are all in there, so a ratio against an exact survival target")
print("would be comparing two different quantities.")
print()

biggest_gap = 0.0
for accuracy in survival:
    gap = abs(survival[accuracy] - terminal_reward[accuracy])
    if gap > biggest_gap:
        biggest_gap = gap
print("largest terminal survival-vs-return gap: %.6f  (should be 0)" % biggest_gap)
print("survival probabilities sum to:", round(sum(shaped_survival.values()), 4),
      "-- one survivor per episode, so this must be about 1")
print("shaped returns sum to:", round(sum(shaped_reward.values()), 4),
      "-- no reason for this to be 1, and it is not")

# %% [markdown]
# ## 3. Count targeting honestly — no decisions is not a rate of zero
#
# **This is the easiest thing in the whole folder to get wrong**, and getting it wrong
# produces a confident false statement rather than a visible error.
#
# The measurement is: when a player fires, how often does it choose the **stronger** of its
# two neighbours? Three rules make that question well posed.
#
# - A decision counts only if the player **fired**. An abstention is not a choice of target.
# - A decision counts only if the two neighbours are **distinct**. With two players left,
#   left and right are the same person, so there is nothing to choose between.
# - A decision counts only if the two neighbours have **different accuracies**. Otherwise
#   "the stronger one" does not name anybody.
#
# So `always_miss` must return **no decisions at all**. If the function instead returned
# 0.0, it would be asserting that the policy always shot the weaker neighbour — a specific,
# false claim about a policy that never fired.


# %%
def targeting_rate(records, accuracy=None):
    """How often a fired shot chose the stronger neighbour.

    Returns (rate, stronger, decisions). `rate` is None when there were no eligible
    decisions, because a rate of 0.0 would be a claim about behaviour that never
    happened.
    """
    stronger = 0
    decisions = 0

    for record in records:
        for decision in record["decisions"]:
            if accuracy is not None and decision["accuracy"] != accuracy:
                continue
            if decision["action"] == MISS:
                continue                                    # did not fire
            if decision["n_alive"] <= 2:
                continue                                    # left and right coincide
            if decision["left_accuracy"] == decision["right_accuracy"]:
                continue                                    # no stronger one to pick

            decisions = decisions + 1
            if decision["action"] == SHOOT_LEFT:
                chosen = decision["left_accuracy"]
                other = decision["right_accuracy"]
            else:
                chosen = decision["right_accuracy"]
                other = decision["left_accuracy"]
            if chosen > other:
                stronger = stronger + 1

    if decisions == 0:
        return None, 0, 0
    return stronger / decisions, stronger, decisions


print("%-18s %12s %10s   %s" % ("policy", "P(stronger)", "decisions", "reading"))
for name, records in [("shoot_strongest", shooting_records),
                      ("random", random_records),
                      ("always_miss", abstain_records)]:
    rate, stronger, decisions = targeting_rate(records)
    if rate is None:
        shown = "no data"
        reading = "never fired, so there is nothing to report"
    else:
        shown = format(rate, ".4f")
        reading = "measured from " + str(decisions) + " fired shots"
    print("%-18s %12s %10d   %s" % (name, shown, decisions, reading))

print()
print("always_miss returns None, NOT 0.0. A rate of 0.0 would assert that it always")
print("shot the weaker neighbour, which is a false statement about a policy that")
print("never fired at all.")

# %% [markdown]
# `shoot_strongest` measures 1.0000 by construction, which is the instrument agreeing with
# the obvious. `random` sits near 0.5 because it picks a side without looking.
#
# **A caution that matters from N=5 onward.** At N=3 every player's two neighbours are the
# other two players, so this single pooled number is the whole story. Above N=3 the
# equilibrium's targeting *depends on how many players remain* — so pooling across
# survivor counts would describe a game nobody plays. Step 5 conditions on it.

# %% [markdown]
# ## 4. The ratio gate — only where an exact answer is on record
#
# Compare measured survival to the exact target; every accuracy must reach 95% of it. The
# 5% margin is chosen against the noise: at 10,000 episodes the standard error of a
# probability near 0.33 is about 0.005, so 5% is comfortably wider.
#
# Applied below to **`shoot_strongest`, not to an agent** — this is the instrument being
# calibrated, not a verdict on anything that learned. At N=3 that heuristic *is* the
# equilibrium, so it should reproduce the exact triple. The first time this gate judges a
# trained policy is step 6.


# %%
# Two different questions, kept apart because they have different answers.
#
# First: for which player counts does an exact answer EXIST? That is a fact about the game,
# recorded here rather than inferred from a claim about where the solver stops. Folder 02
# computes them; at N=9 backward induction reaches a state with no pure Nash equilibrium, so
# beyond N=8 there is nothing to compute at any budget.
EXACT_ANSWERS_EXIST_FOR = {3, 4, 5, 6, 7, 8}

# Second: which of those profiles does THIS folder carry? Only the one it needs. The folder
# stands alone, so anything it uses it holds itself — step 6 carries N=5 for the same reason.
EXACT_PROFILES = {3: EXACT_N3}


def has_exact_answer(n):
    """Whether an exact answer exists for this player count."""
    return n in EXACT_ANSWERS_EXIST_FOR


def ratio_gate(measured, exact, threshold=0.95):
    rows = []
    passed = True
    for accuracy in sorted(exact, reverse=True):
        ratio = measured[accuracy] / exact[accuracy]
        ok = ratio >= threshold
        passed = passed and ok
        rows.append((accuracy, measured[accuracy], exact[accuracy], ratio, ok))
    return passed, rows


passed, rows = ratio_gate(survival, EXACT_N3)
print("  %4s %9s %8s %7s %8s" % ("acc", "measured", "exact", "ratio", ">=0.95"))
for accuracy, measured, exact_value, ratio, ok in rows:
    print("  %4s %9.4f %8.4f %7.3f %8s"
          % (accuracy, measured, exact_value, ratio, "yes" if ok else "NO"))
print()
print("shoot_strongest reaches the exact triple:", passed)
print("  (expected -- at N=3 this heuristic IS the equilibrium)")

# %% [markdown]
# ### Why the gate cannot always be applied
#
# It needs an exact answer to divide by, and beyond a certain field size there is none.
#
# The reason is **not** cost, which is worth being exact about because cost is the intuitive
# guess. The solver enumerates subsets of surviving players, so states grow as $2^n$ — folder
# 02 measured 0.005 s at N=5, 0.022 s at N=6 and 0.103 s at N=7, roughly five times per extra
# player. Carried forward that is still under a minute well past N=8, so waiting is not the
# obstacle.
#
# The obstacle is that backward induction works by selecting a **pure** equilibrium at each
# state, and at N=9 it reaches a state that has none. There is nothing to select, so the solver
# stops. That state does have a mixed equilibrium (folder 02, step 4), but this project's solver
# does not compute mixed ones, so no exact answer is on record above N=8.
#
# So this code asks whether an answer is **on record** for a given N, and returns `None` when
# there is not. Asking for a ratio there is not imprecise, it is **unanswerable**, and a
# function that returned a number anyway would be the worst kind of bug.

# %%
for n, states, seconds in [(5, 26, 0.005), (6, 57, 0.022), (7, 120, 0.103)]:
    print("  N=%d: %4d states, %.3f s" % (n, states, seconds))
print()
print("  exact answers on record for N in", sorted(EXACT_ANSWERS_EXIST_FOR))
print("  profiles carried in this folder:", sorted(EXACT_PROFILES))
print()
for n in [3, 5, 7, 8, 10, 15]:
    if has_exact_answer(n):
        carried = "carried here" if n in EXACT_PROFILES else "carried by the step that needs it"
        print("  N=%2d: exact answer on record, the ratio gate applies -- %s" % (n, carried))
    else:
        print("  N=%2d: no exact answer on record -- the ratio gate must not be used" % n)

# %% [markdown]
# ## 5. The focal-agent protocol — the only option with no exact answer
#
# With no exact target, the comparison becomes: **does the policy do better than a simple
# rule, in the same seat?**
#
# One agent sits at a random position with a random accuracy. Every other seat runs baseline
# B. Count how often that one seat is the sole survivor. Then do it again with baseline B in
# that seat too — the **paired control**.
#
# The control is the whole design. A raw win rate means nothing on its own: a focal seat
# surrounded by `always_miss` opponents wins nearly always, and that says nothing about the
# policy. Only the difference between the two arms does.


# %%
def focal_matchup(game, focal_policy, baseline_policy, episodes, focal_seed=7000):
    """One seat runs focal_policy, the rest run baseline_policy."""
    focal_rng = np.random.default_rng(focal_seed)
    wins = {}
    plays = {}

    for episode in range(episodes):
        game.reset()
        focal_position = int(focal_rng.integers(0, game.n))
        focal_accuracy = round(float(game.accuracies[focal_position]), 4)

        for acting_agent in game.agent_iter():
            observation, reward, termination, truncation, info = game.last()

            # A finished agent is handed to us one last time, so PettingZoo can retire
            # it. Stepping with None is how we say 'nothing to do here'.
            if termination or truncation:
                game.step(None)
                continue

            actor = game.position_of_agent[acting_agent]
            if actor == focal_position:
                action = focal_policy(game, actor)
            else:
                action = baseline_policy(game, actor)
            game.step(action)

        plays[focal_accuracy] = plays.get(focal_accuracy, 0) + 1
        # Sole survivor only. A stagnation split is not a win.
        if len(game.order) == 1 and game.order[0] == focal_position:
            wins[focal_accuracy] = wins.get(focal_accuracy, 0) + 1

    return wins, plays


# PAIRING NOTE. The two arms must face the same situations, or part of any measured
# difference is just a different draw. Two things make that true here:
#
#   * the focal SEAT comes from its own generator, `focal_rng`, so changing the focal
#     policy does not change which seats get chosen;
#   * `seating="ranked"` is used below, so position k always holds the same accuracy.
#     Under `random` seating the game's own generator advances by different amounts in the
#     two arms -- a policy that fires more often consumes more hit rolls -- and the
#     seatings drift apart within a few hundred episodes.


def pooled_rate(wins, plays, accuracies):
    total_wins = 0
    total_plays = 0
    for accuracy in accuracies:
        total_wins = total_wins + wins.get(accuracy, 0)
        total_plays = total_plays + plays.get(accuracy, 0)
    if total_plays == 0:
        return float("nan")
    return total_wins / total_plays


game5 = GruelEnv(n=5, reward="terminal", seating="ranked", seed=1000 + SEED)
EPISODES = 4000

print("Sanity check: with the SAME policy in every seat the focal seat is not special,")
print("so it should win about 1/N. If it does not, the protocol is broken.")
print()
for n, one_game in [(3, game), (5, game5)]:
    wins, plays = focal_matchup(one_game, shoot_strongest, shoot_strongest, EPISODES)
    rate = pooled_rate(wins, plays, list(plays))
    print("  N=%d: focal wins %.4f   1/N = %.4f" % (n, rate, 1 / n))

# %% [markdown]
# Now a real comparison: `random` in the focal seat against a field of `shoot_strongest`,
# versus `shoot_strongest` in that same seat.

# %%
treat_wins, treat_plays = focal_matchup(game5, random_policy, shoot_strongest, EPISODES)
control_wins, control_plays = focal_matchup(game5, shoot_strongest, shoot_strongest,
                                            EPISODES)

print("paired?", treat_plays == control_plays,
      "-- both arms must see the same focal seats")
print()
weakest_two = sorted(control_plays)[:2]
print("weakest two accuracies:", weakest_two)
print("  random in the focal seat        : %.4f"
      % pooled_rate(treat_wins, treat_plays, weakest_two))
print("  shoot_strongest in the same seat: %.4f"
      % pooled_rate(control_wins, control_plays, weakest_two))

# %% [markdown]
# ### Why self-play can never be the gate
#
# If every seat runs the same policy, the survival probabilities **sum to 1 by
# construction**, so the mean win rate is $1/N$ no matter how good the policy is. A whole
# population of learners can look healthy while not one of them can beat a two-line rule.
#
# That is why self-play numbers are descriptive, always.
#
# > The focal/background design is not this project's invention. It is the evaluation
# > method of DeepMind's **Melting Pot** (Leibo et al., ICML 2021; *Melting Pot 2.0*,
# > arXiv:2211.13746), which pairs an environment with "a reference set of co-players (a
# > *background population*)" and scores the focal agent against it. Their stated reason for
# > building it is the reason it is used here: self-play scores "reveal weaknesses not
# > apparent from training performance alone."

# %% [markdown]
# ## 6. Error bars, and what they are not
#
# Two different things get confused here, so both are computed.
#
# A **confidence interval** says how precisely one number is known. A **test** says whether
# two numbers differ. Overlapping intervals mean the difference is not resolved — and that
# is the honest report, not a failure to be argued around.


# %%
def standard_error(probability, episodes):
    return (probability * (1 - probability) / episodes) ** 0.5


def bootstrap_interval(wins, plays, draws=2000, level=0.95, generator=None):
    if plays == 0:
        return (float("nan"), float("nan"))
    if generator is None:
        generator = np.random.default_rng(SEED)
    observations = np.zeros(plays)
    observations[:wins] = 1.0
    means = np.empty(draws)
    for draw in range(draws):
        sample = generator.choice(observations, size=plays, replace=True)
        means[draw] = sample.mean()
    low = float(np.quantile(means, (1 - level) / 2))
    high = float(np.quantile(means, 1 - (1 - level) / 2))
    return low, high


weakest = survival[0.4]
print("weakest player's survival at N=3: %.4f" % weakest)
for episodes in [100, 1000, 10000]:
    print("  at %6d episodes, standard error %.4f"
          % (episodes, standard_error(weakest, episodes)))

print()
wins_weakest = int(round(weakest * len(shooting_records)))
low, high = bootstrap_interval(wins_weakest, len(shooting_records))
print("bootstrap 95%% interval at %d episodes: [%.4f, %.4f]"
      % (len(shooting_records), low, high))
print("exact answer %.4f inside the interval:" % EXACT_N3[0.4],
      low <= EXACT_N3[0.4] <= high)

# %% [markdown]
# ## What must be measured but must never gate
#
# | Quantity | Why it cannot gate |
# |---|---|
# | Miss frequency | A *strategy* fact, not an outcome. Survival-of-the-weakest can arise from targeting alone with no missing at all |
# | Targeting patterns | Same reason — describes *how*, not *what happened* |
# | Self-play survival | Sums to 1 by construction; cannot express "beats a baseline" |
# | Anything under `shaped` compared to an exact target | Different objective, different equilibria |
# | Any single seed | Several seeds minimum for any claim that separates two things |
#
# The last row is a rule rather than a caution: a difference visible in one seed and absent
# when pooled was never a difference.

# %%
print("miss frequency by accuracy:")
for name, records in [("shoot_strongest", shooting_records),
                      ("random", random_records)]:
    counts = {}
    misses = {}
    for record in records:
        for decision in record["decisions"]:
            shooter_accuracy = decision["accuracy"]
            counts[shooter_accuracy] = counts.get(shooter_accuracy, 0) + 1
            if decision["action"] == MISS:
                misses[shooter_accuracy] = misses.get(shooter_accuracy, 0) + 1
    parts = []
    for accuracy in sorted(counts, reverse=True):
        parts.append(str(accuracy) + ": " + format(misses.get(accuracy, 0) / counts[accuracy], ".4f"))
    print("  " + name.ljust(17) + "  " + "   ".join(parts))
print()
print("Zero for shoot_strongest, about a third for random -- both by construction.")
print("In the equilibrium the theory predicts near-zero abstention too, so a low miss")
print("rate is the prediction being met, not a shortfall.")

# %% [markdown]
# ## Reproducibility

# %%
def measure_twice():
    fresh = GruelEnv(n=3, reward="terminal", seating="ranked", seed=1000 + SEED)
    records = play_episodes(fresh, shoot_strongest, 500)
    measured = survival_by_accuracy(records)
    return {a: round(v, 6) for a, v in measured.items()}


first_run = measure_twice()
second_run = measure_twice()

print("run 1:", first_run)
print("run 2:", second_run)
print()
print("identical:", first_run == second_run)
