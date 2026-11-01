# Step 1 — REINFORCE at N=2

**The smallest possible policy gradient, on the smallest possible game.**

> **This folder stands alone.** It has its own copy of `gruel_env.py`, its own
> `requirements.txt`, and imports nothing from any other folder. Copy it anywhere and it runs.

Player count here: **N=2**.

## Why this step exists

**Why start here.** REINFORCE is the whole policy-gradient idea in one line: push up the
log-probability of actions that preceded a good return, push down the ones that did not. Nothing else in
this folder makes sense until that line is on the screen and working.

**Why N=2.** In a two-player duel the circle has collapsed: left and right are the *same* opponent. So
the three actions reduce to two real choices — shoot, or deliberately miss. That is the smallest game in
which the abstention question exists at all, and it means the whole episode is short enough to print.

## What it contains

- The policy network: four observed numbers in, three action logits out.
- One episode played by hand, printed decision by decision.
- The REINFORCE loss, $-\log \pi(a \mid s) \times R$, and why the minus sign is there.
- Training, and the survival rate settling where it should.

## What number to expect

If both players always shoot, the duel has an exact answer. One player is drawn at random each
step, so with probability $1/2$ the first player acts and hits with probability $p$; if nobody hits, the
game is in exactly the state it started in, which lets the same unknown appear on both sides and be
solved for:

$$W(p, q) = \frac{p}{p + q}$$

With accuracies 0.4 and 0.8 that gives **0.3333 for the weak player and 0.6667 for the strong one** — the
stronger player wins twice as often. The notebook computes both before training, then measures the
trained policy over 20,000 evaluation episodes and prints the ratio of measured to exact.

**At N=2 there is no strategy to find** — whoever shoots more accurately wins more often. That is the
point of this step: a correct answer that needed no cleverness, so that a later step showing something
surprising cannot be dismissed as a bug.

The policy should also learn to **almost never abstain**, since giving a duelling opponent free shots is
suicide. Note that `shoot_left` and `shoot_right` point at the same person here, so only their **total**
is meaningful; how the policy splits weight between the two is arbitrary.

**What this folder actually measured**, at `SEED = 0` over 20,000 evaluation episodes:

| Player | Measured | Exact | Ratio |
|---|---|---|---|
| weak, 0.4 | 0.3301 | 0.3333 | 0.990 |
| strong, 0.8 | 0.6700 | 0.6667 | 1.005 |

Abstention fell to **0.0007** for the weak player and **0.0005** for the strong one, and the split between
`shoot_left` and `shoot_right` came out near 0.48 / 0.52 — arbitrary, as expected, since both point at the
same opponent. All 20,000 episodes ended with a sole survivor: none stagnated.

## What to watch in the output

**How much of each batch does nothing.** Under the `terminal` reward the loss is
$-\log \pi(a \mid s) \times R$ with $R$ equal to 1 for the winner and 0 for everybody else. So every
decision made by the player who lost is multiplied by zero and contributes **no gradient at all**. The
notebook measures this over 200 untrained episodes and finds **0.280** of decisions discarded. Below half,
and worth understanding why: the killing shot is always the winner's, and a player drawn more often is
both more likely to win and to have made more decisions, so the surviving player's decisions outnumber
the loser's. Only those are pushed up. That is why the curve wobbles, and it is the weakness step 2
addresses.

**This loss can never go negative.** A probability is at most 1, so $\log \pi \le 0$, so
$-\log \pi \ge 0$; and $R$ is never negative here. A negative number would mean something is wrong.
That changes in step 2, where the advantage can be negative by design.

## Reproducibility

The notebook ends by running the same thing twice from the same seed and printing whether the
two results are identical. A step that cannot reproduce itself cannot support a claim.

## Where this sits

- Next: [`../step2-ppo/`](../step2-ppo/) — The same game as step 1, so the only thing that changes is the algorithm.
- The whole investigation: [`../README.md`](../README.md)
