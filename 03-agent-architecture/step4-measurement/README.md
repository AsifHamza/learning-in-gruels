# Step 4 — The measurement machinery

**Build the instruments, and check them against answers already known.**

> **This folder stands alone.** It has its own copy of `gruel_env.py`, its own
> `requirements.txt`, and imports nothing from any other folder. Copy it anywhere and it runs.

Player count here: **N=3**.

## Why this step exists

**Why this is a step and not a utility file.** Every claim the rest of the folder makes
depends on measuring the right thing. An instrument that is subtly wrong produces confident, wrong
findings — which is worse than no findings. So the instruments are built here, against **heuristic
policies whose behaviour is known in advance**, before they are ever pointed at a trained agent.

**Why classification comes before reading any number.** Two equilibria exist in this variant: one where
players shoot, and one where they abstain and the game stalls. The analytical targets describe the
*shooting* one. So the stagnation rate has to be read **first** — if a run stalled, its survival numbers
describe a different game and comparing them to the targets is meaningless.

## What it contains

- Survival by accuracy rank, counted from who actually survived rather than from rewards.
- The stagnation and hard-cap rates, and the rule that they are read first.
- The **ratio gate**: measured ÷ exact at each rank, with a threshold — usable only where an exact answer is on record.
- The **focal-agent protocol**, the only option once no exact answer is on record.
- Standard errors and bootstrap intervals, and what they do *not* tell you.
- What must be measured but must never be allowed to gate.

## What number to expect

Each heuristic behaving exactly as its name says. An instrument that disagrees with a policy whose
answer is obvious is a broken instrument.

**Classification.** `always_miss` stagnates in every episode; the other two never do.

| Policy | Stagnation | Hard cap | Class |
|---|---|---|---|
| `shoot_strongest` | 0.0000 | 0.0000 | shooting |
| `always_miss` | **1.0000** | 0.0000 | non_shooting |
| `random` | 0.0000 | 0.0000 | shooting |

**Targeting**, and this is the row the whole folder exists to get right:

| Policy | P(shoot the stronger neighbour) | Decisions | Reading |
|---|---|---|---|
| `shoot_strongest` | 1.0000 | 5002 | 1.0 by construction |
| `random` | 0.5093 | 5028 | picks a side without looking |
| `always_miss` | **no data** | **0** | never fired, so there is nothing to report |

**Survival against reward.** Under `terminal` the two are the same number — the largest gap measured is
**0.000000**. Under `shaped` they are not: the same three players collect mean returns of −0.3255,
−0.2709 and −0.1969, which sum to **−0.7933**. Survival probabilities must sum to about 1 because there
is one survivor per episode; returns have no such constraint, and comparing them to an exact survival
target would be comparing two different quantities.

**The ratio gate**, applied to `shoot_strongest` rather than to anything that learned, because at N=3
that heuristic *is* the equilibrium and so should reproduce the exact triple:

| Accuracy | Measured | Exact | Ratio |
|---|---|---|---|
| 0.8 | 0.2990 | 0.29630 | 1.009 |
| 0.6 | 0.3307 | 0.33333 | 0.992 |
| 0.4 | 0.3703 | 0.37037 | 1.000 |

**The focal protocol's sanity check.** With the same policy in every seat the focal seat is not special,
so it must win about $1/N$: measured **0.3257** at N=3 against 0.3333, and **0.2065** at N=5 against
0.2000. If that did not hold, the protocol would be measuring the seat rather than the policy.

## What to watch in the output

**That `always_miss` yields no decisions rather than zero.** The difference between "no data" and "a
measurement of zero" is the difference between honest and misleading, and it is the easiest thing in the
whole folder to get wrong. A rate of 0.0 would assert that the policy *always shot the weaker
neighbour* — a specific, confident, false claim about a policy that never fired. So the function returns
`None` and a decision count of 0, and the caller has to deal with it.

Three rules make the targeting question well posed, and each one excludes decisions that would otherwise
pollute the average: the player must have **fired**; its two neighbours must be **distinct**, which rules
out every two-player state where left and right are the same person; and they must have **different
accuracies**, or "the stronger one" names nobody.

**That no gate is applied to an agent here.** This step builds the instruments and calibrates them. The
ratio gate is demonstrated on a heuristic; the first time it judges something that learned is step 6.

## Reproducibility

The notebook ends by running the same thing twice from the same seed and printing whether the
two results are identical. A step that cannot reproduce itself cannot support a claim.

## Where this sits

- Previous: [`../step3-mappo/`](../step3-mappo/) — The centralised critic. This is the architecture the capstone is about.
- Next: [`../step5-targeting/`](../step5-targeting/) — The diagnostic that survival numbers cannot replace.
- The whole investigation: [`../README.md`](../README.md)
