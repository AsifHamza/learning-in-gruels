# Step 3 — MAPPO at N=3

**The centralised critic. This is the architecture the capstone is about.**

> **This folder stands alone.** It has its own copy of `gruel_env.py`, its own
> `requirements.txt`, and imports nothing from any other folder. Copy it anywhere and it runs.

Player count here: **N=3**.

## Why this step exists

**Why the game changes now.** At N=2 there is no question of *who* a state is good for —
there is one opponent and the game is symmetric. At N=3 the circle is a real triangle, three players have
three different accuracies, and "how good is this position" has three different answers at once. That is
the question a shared critic cannot answer.

**Why MAPPO.** Two facts about this game break PPO's critic:

- **The reward is per-agent and almost always zero.** One player wins and takes +1; everyone else takes
  nothing. A critic estimating the value of a state must be told *for whom*.
- **The actor is drawn at random, with replacement.** One agent may act eight times in an episode while
  another acts once. Each agent's experience is scattered irregularly through one shared history.

Together those mean a critic seeing only the acting agent's four numbers cannot tell a return that agent
**earned** from one it **inherited** because someone else happened to die first.

MAPPO's answer is a critic that takes the global state **and which agent is being scored** — $V_\phi(s,
i)$. Training is centralised, execution stays decentralised: the critic sees everything while learning,
and the policy still sees only its own four numbers when it plays.

**Whether this is needed is tested in step 10, not assumed here.** The argument above says a local critic is at a disadvantage; step 10 measures how much. At N=3 and N=5 a critic given only the actor's four numbers passes the same gates, and at N=7 the centralised critic does measurably better without passing.

## What it contains

- The global state vector, and the one-hot saying whose value is being asked for.
- Why the *policy* is deliberately not given that state — demonstrated rather than asserted:
  the trained **actor accepts an N=5 observation unchanged**, because its input is four
  numbers whatever the player count, while the critic *refuses* it, because its input is
  $2N + N$ — 9 at N=3 against 15 at N=5. That asymmetry is what "centralised training,
  decentralised execution" actually means, and it is what lets later steps change N without
  rebuilding the policy.
- Per-agent credit: each agent's own decisions carry its own outcome, kept in separate lists
  during collection because the actor is drawn at random and one agent may act four times
  while another acts once.
- That **the update itself needed no change at all** for three agents. The algorithm did not
  become multi-agent; the data handling did.
- Training at N=3, and survival by accuracy compared with the exact answer.

## What number to expect

Survival by accuracy in the neighbourhood of the exact triple **0.29630 / 0.33333 / 0.37037**
for accuracies 0.8 / 0.6 / 0.4 — that is, **the weakest player surviving most often**, which is the first
genuinely counter-intuitive result in the project. The reason is structural rather than clever: the two
better shots are each other's most dangerous opponent, so they spend their turns removing each other
while the weakest is left comparatively alone.

At a notebook budget expect the ordering to be right and the values to be close, not exact. **No gate is
applied here** — the instrument that turns a ratio into a pass or fail is built in step 4, and the first
real pass/fail is step 6.

**What this folder measured**, from 38,400 episodes of self-play and 20,000 evaluation episodes at
`SEED = 0`:

| Accuracy | Measured | Exact | Ratio |
|---|---|---|---|
| 0.4, the weakest | **0.3639** | 0.37037 | 0.983 |
| 0.6 | 0.3317 | 0.33333 | 0.995 |
| 0.8, the strongest | 0.3044 | 0.29630 | 1.027 |

The ordering is right and every value is within 3% of exact. Stagnation was **0.0000** and abstention
**0.0005** of decisions.

## What to watch in the output

**The ordering, before the values.** Weakest-highest is the claim. Three numbers that are individually
close but in the wrong order have failed, and the notebook therefore prints the ordering test first.

Nothing in an agent's observation announces its global rank — it sees three accuracies and cannot tell
whether 0.6 is the best or the worst in the game. So getting the ordering right is a stronger result than
the individual numbers suggest.

**The equilibrium class, before either.** This game has two equilibria: one where players shoot and one
where everybody abstains forever. The exact triple describes the **shooting** one, so a run that stalls is
not a run that disagrees with the answer — it is a run measuring a different game. The notebook classifies
from the measured stagnation rate first and says so, rather than assuming it.

## Reproducibility

The notebook ends by running the same thing twice from the same seed and printing whether the
two results are identical. A step that cannot reproduce itself cannot support a claim.

## Where this sits

- Previous: [`../step2-ppo/`](../step2-ppo/) — The same game as step 1, so the only thing that changes is the algorithm.
- Next: [`../step4-measurement/`](../step4-measurement/) — Build the instruments, and check them against answers already known.
- The whole investigation: [`../README.md`](../README.md)
