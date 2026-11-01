# %% [markdown]
# # Step 1 — The duel, and the cancellation that makes everything easy
#
# Folder 01 measured survival by playing 100,000 episodes. That works, but it is never
# exact — there is always a wobble of about 0.0015 left over, and to halve it you need four
# times the episodes.
#
# This folder computes the answer **exactly**. No episodes, no randomness, no error bars.
# And it needs no libraries at all — the whole thing is arithmetic.
#
# We start with two players, because the two-player case contains the one idea that makes
# the rest of the folder short.

# %% [markdown]
# ## 1.1 The problem
#
# Two players, accuracies $p$ and $q$. Each step, one of them is picked at random — fair
# coin — and fires at the other. If it hits, they win.
#
# What is the probability that the player with accuracy $p$ survives?
#
# The obvious approach is to add up every way it could happen. $p$ wins on step 1, or the
# first two shots miss and $p$ wins on step 3, or... an infinite series. That works, and it
# is horrible.
#
# Here is the better way.

# %% [markdown]
# ## 1.2 Writing down the equation
#
# Let $W$ be the probability that $p$ eventually wins. From the start of any step:
#
# - With probability $\tfrac12$, **$p$ shoots**. Then with probability $p$ it hits and $p$ has
#   won. With probability $1-p$ it misses, and we are back where we started — the situation
#   is *identical*, so the chance of winning from there is $W$ again.
# - With probability $\tfrac12$, **$q$ shoots**. With probability $q$ it hits and $p$ has
#   lost. With probability $1-q$ it misses, and again we are back to $W$.
#
# Putting that together:
#
# $$W = \tfrac{1}{2}\Big[p + (1-p)W\Big] + \tfrac{1}{2}\Big[(1-q)W\Big]$$
#
# The reason this is legal is that the game has **no memory**. Nothing about the state
# changes when a shot misses — same two players, same accuracies, same whose-turn rules. So
# the value of being in that situation is the same number as before.

# %% [markdown]
# ## 1.3 Solving it
#
# It is one equation with $W$ on both sides. Collect the $W$ terms:
#
# $$W = \tfrac{p}{2} + \tfrac{(1-p)}{2}W + \tfrac{(1-q)}{2}W$$
#
# $$W - \tfrac{(1-p) + (1-q)}{2}W = \tfrac{p}{2}$$
#
# $$W\left[1 - \frac{2 - p - q}{2}\right] = \frac{p}{2}
#   \qquad\Longrightarrow\qquad W \cdot \frac{p + q}{2} = \frac{p}{2}$$
#
# $$\boxed{\;W(p, q) = \frac{p}{p + q}\;}$$
#
# **The coin flip vanished. The waiting time vanished.** Only the *ratio* of the two
# accuracies survives. Whether the duel takes 2 steps or 200, and whether players shoot
# often or rarely, makes no difference to who wins — only their relative deadliness does.
#
# This cancellation is not a trick special to duels. It is the reason the whole solver in
# step 2 fits on a page.

# %%
def duel(p, q):
    """Probability that the player with accuracy p survives a duel against q."""
    return p / (p + q)


print("p=0.8 vs q=0.4 ->", duel(0.8, 0.4))
print("p=0.4 vs q=0.8 ->", duel(0.4, 0.8))
print("p=0.6 vs q=0.4 ->", duel(0.6, 0.4))
print("p=0.5 vs q=0.5 ->", duel(0.5, 0.5))
print()
print("adds to 1:", duel(0.8, 0.4) + duel(0.4, 0.8))

# %% [markdown]
# ## 1.4 Checking it against brute force
#
# The formula came out of algebra, so it deserves a second opinion. Instead of simulating,
# we can compute the same thing by **iteration**: start with a guess for $W$, plug it into
# the right-hand side, and repeat. If the equation is right, the guess settles down onto the
# answer.
#
# This is worth doing once by hand because it is exactly what the step-2 solver does, only
# in one dimension.

# %%
def duel_by_iteration(p, q, rounds=200):
    """Solve W = 0.5*(p + (1-p)W) + 0.5*(1-q)W by repeatedly plugging W back in."""
    guess = 0.0

    for step in range(rounds):
        p_shoots = 0.5 * (p + (1 - p) * guess)
        q_shoots = 0.5 * ((1 - q) * guess)
        guess = p_shoots + q_shoots

    return guess


print("accuracies      formula      iteration     difference")
for p, q in [(0.8, 0.4), (0.6, 0.4), (0.4, 0.8), (0.9, 0.1), (0.5, 0.5)]:
    formula = duel(p, q)
    iterated = duel_by_iteration(p, q)
    print(f"  {p} vs {q}     {formula:.10f}   {iterated:.10f}   {abs(formula - iterated):.2e}")

print()
print("Same answer to ten decimal places. The algebra is right.")

# %% [markdown]
# ## 1.5 The same thing again, without the coin flip
#
# There is an even shorter way to see it, and it generalises directly to N players.
#
# Forget the timing entirely and ask only: **who lands the first hit?** Per step,
#
# - $p$ lands a hit with probability $\tfrac12 p$
# - $q$ lands a hit with probability $\tfrac12 q$
# - nothing happens otherwise
#
# "Nothing happens" just means we try again, and it does not favour anyone. So conditional on
# *someone* being hit, the chance it was $p$'s shot is
#
# $$\frac{\tfrac12 p}{\tfrac12 p + \tfrac12 q} = \frac{p}{p+q}$$
#
# Same answer, no algebra. The recipe is:
#
# > **Throw away the steps where nothing happens. Weight each thing that *can* happen by how
# > fast it happens.**
#
# Those weights — accuracy of the shooter — are called **hazards**. Everything in step 2 is
# this idea applied repeatedly.

# %%
def duel_by_hazard(p, q):
    """Same answer, framed as 'who fires the first successful shot'."""
    hazard_p = p
    hazard_q = q
    total_hazard = hazard_p + hazard_q
    return hazard_p / total_hazard


print("hazard version matches formula:", duel_by_hazard(0.8, 0.4) == duel(0.8, 0.4))

# %% [markdown]
# ## 1.6 Why 1/3 and 2/3 turn up in folder 03
#
# A useful case to remember. Folder 03 will use the N=2 duel as its quick check that
# training works, because it converges fast and the right answer is a simple fraction.
#
# With accuracies 0.4 and 0.8:
#
# $$W(0.4, 0.8) = \frac{0.4}{1.2} = \frac13, \qquad W(0.8, 0.4) = \frac{0.8}{1.2} = \frac23$$

# %%
print("weak player (0.4) survives:", duel(0.4, 0.8), "= 1/3 ?", abs(duel(0.4, 0.8) - 1/3) < 1e-12)
print("strong player (0.8) survives:", duel(0.8, 0.4), "= 2/3 ?", abs(duel(0.8, 0.4) - 2/3) < 1e-12)

# %% [markdown]
# Note that in a **duel** there is no survival-of-the-weakest. The better shot simply wins
# more often, in exact proportion to how much better it is.
#
# Survival-of-the-weakest needs at least **three** players, because it depends on opponents
# choosing *each other* as targets. It is a fact about targeting, not about shooting.

# %% [markdown]
# ## 1.7 No seeding needed
#
# Worth saying explicitly: there is no `SEED` in this notebook and there will not be one in
# the others either. Nothing here is random. Exact computation gives the same answer every
# time by construction, not by seeding — which is the main reason this folder is worth having
# at all.

# %% [markdown]
# ## What to take from step 1
#
# - $W(p,q) = \dfrac{p}{p+q}$: in a duel only the **ratio** of accuracies matters.
# - The self-reference "if we both miss we are back where we started" **cancels**, because
#   the game has no memory. That collapse is what makes the exact solver small.
# - Equivalently: ignore the steps where nothing happens, and weight outcomes by **hazard**
#   (the shooter's accuracy).
# - $W(0.4, 0.8) = 1/3$ — the number folder 03 uses as its fast sanity check.
# - A duel has no survival-of-the-weakest. That needs three players and their targeting
#   choices.
#
# Next: `step2_truel_n3.py` applies the hazard idea to three players and lands on 8/27, 9/27,
# 10/27.
