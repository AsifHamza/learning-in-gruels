# Step 7 — Game variant: seating

**One dial moved. Same agent, same player count, different answer.**

> **This folder stands alone.** It has its own copy of `gruel_env.py`, its own
> `requirements.txt`, and imports nothing from any other folder. Copy it anywhere and it runs.

Player count here: **N=5**.

## Why this step exists

**Why seating is the first variant to test.** It is the cleanest possible experiment in this
folder: the agent is unchanged, N is unchanged, the reward is unchanged, and the *only* difference is
which accuracy sits where. If the answer moves, seating moved it.

**Why it should move.** Under **ranked** seating the two strongest players are always adjacent, so they
spend the early game shooting each other and the weakest player is structurally protected. Under
**random** seating that adjacency is redrawn every episode, so the protection becomes a lottery — and an
agent seeing only its two neighbours cannot know which arrangement it is in.

That last clause is the interesting part. The agent has no way to observe the global layout, so it cannot
adapt to it. What it learns instead is an average over layouts.

## What it contains

- The same MAPPO agent trained twice at N=5, changing only `seating`.
- Survival by accuracy rank for both, side by side against the ranked exact profile.
- Targeting for both, which is where the difference shows most sharply.
- A note on why the exact profile is *only* the right comparison for the ranked run.

## What number to expect

**Both arms train cleanly** — `stagnation rate: 0.0000` in all six runs — and the ranked arm
reproduces the exact profile: `0.2431` / `0.2125` / `0.1855` / `0.1610` / `0.1979`, ratios 0.974 / 1.009 /
1.032 / 0.994 / 0.999.

**The random arm measures `0.3894` for the weakest player** — that is `1.947` times an even share,
against `1.215` for the ranked arm. **The weakest does better when the seats are shuffled**, which is the
opposite of what a reader would predict, and the reason is not visible in the survival numbers at all.

**Targeting is what explains it.** The two arms learned different policies:

| | targeting by rank, weakest → strongest | spread |
|---|---|---|
| ranked | `0.9999` / `0.3790` / `0.3142` / `0.3745` / `0.9998` | `0.6857` |
| random | `0.8855` / `0.9344` / `0.9451` / `0.9463` / `0.9353` | `0.0608` |

The ranked arm has the equilibrium's U-shape. The random arm is flat and high — the signature of
`shoot_strongest`. Measured against two reference policies the notebook plays itself, the random arm sits
`0.0707` from `shoot_strongest`, while the ranked arm's nearest reference is `0.2873` away and is
therefore reported as matching **neither**.

**The equilibrium check settles it**, on the axis where an exact answer exists — how targeting changes as
the circle contracts:

| | 5 alive | 4 alive | 3 alive | worst gap from exact |
|---|---|---|---|---|
| exact equilibrium | 0.40 | 0.50 | 1.00 | — |
| ranked | `0.4038` | `0.5102` | `0.9763` | `0.0237` |
| random | `0.8833` | `0.9242` | `0.9662` | `0.4833` |

**The comparison with 0.40 / 0.50 / 1.00 is a ranked-seating answer, so the random arm's gap from it is not evidence by itself.** The same policy could measure differently simply because the seats differ. The notebook settles this by playing every trained model in the *other* seating:

| trained, then played in | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 |
|---|---|---|---|---|---|
| ranked, ranked | `0.9998` | `0.3815` | `0.3121` | `0.3706` | `0.9996` |
| random, ranked | `0.9939` | `0.9992` | `0.9982` | `0.9967` | `0.9916` |
| ranked, random | `0.5587` | `0.6190` | `0.6451` | `0.6336` | `0.6044` |
| random, random | `0.8833` | `0.9325` | `0.9480` | `0.9467` | `0.9360` |

Seated by rank, the random-trained models are `0.0041` from `shoot_strongest` and fire at the stronger neighbour `0.9989` / `0.9982` / `0.9917` of the time with five, four and three alive, a worst gap of `0.5989` from the ranked equilibrium. The ranked-trained models, played in the same ranked game, give `0.3979` / `0.5118` / `0.9763`. So the two arms learned different policies; it is not one policy measured in two games. What the random arm's policy is *worth* under random seating cannot be checked, because no exact answer is on record for an agent that cannot see the arrangement.

**So the finding is not that random seating removes the weakest player's protection.** It is that the
agent **stopped finding the equilibrium** and settled on a cruder rule — and that cruder rule is *even
kinder* to the weakest, because it concentrates everybody's fire on whoever is strongest. The notebook
confirms this by playing `shoot_strongest` itself under random seating: it gives the weakest `0.3952`,
against the agent's `0.3894` — a ratio of `0.985`. Meanwhile `coin_flip`, a policy that discriminates on
nothing, gives the weakest only `0.1285`.

## What to watch in the output

**That the ranked run is the one compared to the exact numbers.** Holding the random run against a
ranked target is the easiest mistake to make here, and it would make a correct run look broken. In this
notebook the comparison function **refuses** — asking it for exact ratios on the random arm raises an
error that states why, because random seating re-deals the seats every episode and therefore measures an
average over all 120 arrangements rather than the one the solver answered.

**That "only one dial moved" is checked rather than claimed.** The notebook lists what is held fixed
and then compares the two environment objects attribute by attribute, printing the count of differences.
It must be exactly one.

**That no reference is named unless it is close.** Both reference policies are flat, so a policy whose
targeting swings across ranks will be nearest to one of them by arithmetic alone while behaving like
neither. A match is only reported within a stated threshold of `0.10`; otherwise the answer is
*neither*, which is what the ranked arm gets.

**That the survival numbers alone would have misled here.** This is the clearest case in the folder
for why step 5 exists: the weakest player doing *better* under shuffled seats reads as "shuffling helps
the weak" and means nothing of the kind.

## Reproducibility

The notebook ends by running the same thing twice from the same seed and printing whether the
two results are identical. A step that cannot reproduce itself cannot support a claim.

## Where this sits

- Previous: [`../step6-n5-gate/`](../step6-n5-gate/) — The first real pass or fail, against a profile that is not monotone.
- Next: [`../step8-reward-profile/`](../step8-reward-profile/) — Does the finding survive the reward that produced it?
- The whole investigation: [`../README.md`](../README.md)
