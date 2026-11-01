# Deliverable 01 — The Environment

**What this deliverable is:** the game itself, in code. The rules, the graph, the turn order, the
shooting, the rewards, four simple policies to compare against, and a picture of it running.

Nothing here learns anything — that is folder 03. This folder is the thing the learning will
happen *inside*, and everything later depends on it being right, which is why it ends by checking
itself against an answer known exactly.

---

## The game

N players sit on a circle. Each has an **accuracy** — the chance a shot they fire hits. Accuracies
are spread evenly over [0.4, 0.8], so at N=3 they are exactly 0.4, 0.6, 0.8.

Every step:

1. One living player is picked **uniformly at random** — not in turn order, and the same player can
   be picked twice in a row.
2. They shoot their left neighbour, shoot their right neighbour, or deliberately miss.
3. If they shot, a random roll against their accuracy decides whether it hits.
4. A hit removes the target and the circle **contracts** — the victim's two former neighbours
   become neighbours of each other.

Last player alive scores +1, everyone else 0. If everyone keeps deliberately missing, the game is
declared stuck and the +1 is split among the survivors.

You may only shoot someone **adjacent** to you. That restriction, plus contraction, is what makes
this a graph problem rather than a free-for-all.

---

## What to expect

The number this whole folder is built around. At N=3, accuracies 0.8 / 0.6 / 0.4, everyone
shooting their strongest neighbour:

| Accuracy | Exact survival | As a fraction |
|---|---|---|
| 0.8 (best shot) | 0.29630 | 8/27 |
| 0.6 | 0.33333 | 9/27 |
| 0.4 (worst shot) | **0.37037** | 10/27 |

**The worst shot is the most likely to survive.** That is what the thesis is about, and it happens
for a reason with no cleverness in it: the two good shots spend their turns shooting each other,
because each is the other's biggest threat.

If any notebook prints numbers missing those three by more than about three standard errors,
something is broken. Every notebook that measures survival prints the comparison for you.

---

## The steps

Read them in order — each assumes the one before. Every step folder has its own README with the
exact commands and the exact output to expect.

| Folder | What it teaches | Runtime |
|---|---|---|
| [`step1-circle-graph/`](step1-circle-graph/) | The shape. Accuracies, seating, neighbours, and what contraction actually is | instant |
| [`step2-plain-game/`](step2-plain-game/) | The whole game in plain Python, no framework. Ends by hitting 8/27, 9/27, 10/27 | ~5 s |
| [`step3-pettingzoo-env/`](step3-pettingzoo-env/) | The same game wrapped in PettingZoo. What the framework adds, and what it demands back | ~15 s |
| [`step4-heuristics/`](step4-heuristics/) | The four baseline policies, measured with error bars, plus the focal-agent comparison | ~35 s |
| [`step5-render/`](step5-render/) | Drawing it. Why dead players stay in their seats. Includes `play.py`, a live window | ~10 s |

Total runtime for the whole folder is a bit over a minute.

---

## Running any of it

The virtualenv lives here, at the folder level, and is shared by all five steps.

```bash
cd 01-environment
python3 -m venv .venv                            # first time only
./.venv/bin/pip install -r requirements.txt      # first time only
./.venv/bin/jupyter lab                          # then navigate into a step folder
```

Or from inside a step folder, referring back up to the venv:

```bash
cd step2-plain-game
../.venv/bin/jupyter lab step2_plain_game.ipynb
```

The notebooks are saved **with their outputs already in them**, so you can read the results
without running anything. Re-running reproduces them exactly — every file sets `SEED = 0`.

Each step folder holds two files with the same content. Read the **`.ipynb`**. The `.py` is the
plain-text form of the same notebook, paired by jupytext, which is what gets edited and converted
— ignore it unless you want to diff it in git.

---

## Changing N

Every notebook has `N` near the top. Change it and re-run the whole notebook.

Two things to know first.

**At N=3 the graph does nothing.** A triangle is both a circle and a complete graph, so everyone
can already shoot everyone. That is exactly why N=3 is the right place to start — it isolates the
game from the graph. The adjacency restriction only bites at N=4 and above.

**The exact answer only exists at N=3.** The 8/27 comparison cells check themselves with
`if N == 3` and quietly skip otherwise. For other player counts you can still measure survival,
you just have nothing to check it against — until folder 02, which computes the answer for any N
up to about 8.

---

## Things worth knowing before reading the code

**A deliberate miss and a shot that missed are different events.** Both look like nothing
happened. One is a choice, the other is bad luck. Only the deliberate one advances the counter
that detects a stuck game — a fired shot resets it, because the game is still moving. Getting this
backwards changes the game without breaking it.

**Survival is counted by accuracy, never by position.** Seating is reshuffled every episode, so
"position 0 won 30% of the time" is meaningless. The question is always what the 0.4 player did,
wherever they were sitting.

**Rewards are added up, not wins counted.** A stuck episode has no winner — everyone gets 1/N.
Counting winners would silently discard those episodes and the probabilities would not add to 1.

**One agent acts per step.** This is why the environment has to be PettingZoo's AEC style rather
than the parallel style, and it is the root of the hard part of folder 03: with agents acting at
irregular intervals there is no single tidy trajectory to learn from.

---

## Four rules this folder exists to teach

- **On a contracting graph, "my neighbour" is only true at one instant.** Read the target *before*
  removing anyone.
- **Run someone else's test of your interface.** PettingZoo's `api_test` checks things your own tests
  will not think to.
- **Look at the picture.** `-pi/2` puts position 0 at the bottom, because matplotlib measures from the
  positive x-axis. No amount of reading the code reveals that.
- **Label figures to enough precision to distinguish the values.** One decimal makes 0.56 and 0.64
  identical.

## What this folder deliberately does not do

- **No learning.** No neural networks, no training. Folder 03.
- **No exact solver.** This folder *measures* survival by playing many episodes, which always
  leaves sampling error. Folder 02 computes it exactly.
- **No configuration files.** Parameters are literals at the top of each file, so there is never a
  question of where a number came from.
- **No shared module.** Each notebook contains its own copy of the game — duplication on purpose,
  so you can read any one file top to bottom without jumping elsewhere.
