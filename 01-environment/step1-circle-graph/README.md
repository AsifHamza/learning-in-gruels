# Step 1 — The circle, and what contraction means

## What this step is

The **shape** of the game, and nothing else. No turns, no shooting, no winning.

N players sit on a circle. Each has two neighbours, a left and a right, and those are the only
two people they are ever allowed to shoot. When somebody dies they are removed, and their two
former neighbours become neighbours of each other — the circle **contracts**.

The one idea worth taking away is how little code that needs. You might expect a graph class
with a list of edges, and code to delete edges and add new ones on a death. You need none of
it. Keep a **list of positions in circular order**; your neighbours are whoever is beside you
in the list; and to kill someone, remove them from the list. Their neighbours are now
automatically adjacent.

Contraction is not something the code does. It is something that happens because of how the
data is stored.

## How to run it

From this folder:

```bash
../.venv/bin/jupyter lab step1_circle_graph.ipynb
```

Or open Jupyter at the folder level and navigate:

```bash
cd ..
./.venv/bin/jupyter lab
```

The notebook is saved with its outputs already in it, so you can read it without running
anything. To re-run: *Kernel → Restart Kernel and Run All Cells*.

Both files here hold the same content. Read the **`.ipynb`**. The `.py` is the plain-text
form of the same notebook, which is what gets edited and converted — ignore it unless you want
to diff it in git.

## What you should see

Runs instantly. Seven cells produce output.

Accuracies and seating:

```
accuracies, weakest first: [0.4 0.6 0.8]
ranked seating : [0.4 0.6 0.8]
random seating : [0.8 0.4 0.6]
```

Who can shoot whom at N=3, and the proof that the graph is not restricting anything yet:

```
position 0 (accuracy 0.8) can shoot left=2 or right=1
opponents:            2
reachable neighbours: 2
=> everyone can reach everyone. The graph is not restricting anything.
```

A contraction, before and after, on a 6-player circle:

```
BEFORE
  neighbours of 1: (0, 2)
  neighbours of 3: (2, 4)
  are 1 and 3 neighbours? False

player 2 has been eliminated

AFTER
  neighbours of 1: (0, 3)
  neighbours of 3: (1, 4)
  are 1 and 3 neighbours? True
  the contraction joined 1 and 3
```

The circle surviving six deaths in a row with every living player keeping exactly 2 neighbours:

```
killed 5 -> 7 alive, order [0, 1, 2, 3, 4, 6, 7], degrees [2, 2, 2, 2, 2, 2, 2]
...
Every living player kept exactly 2 neighbours. Still a circle.
```

And the seeding check:

```
same seed gives the same seating: True
```

If that last line says `False`, or any degree is not 2, something is broken.

## What to look at

**The N=3 special case.** At three players the circle *is* a complete graph — a triangle. So
everyone can already shoot everyone and the adjacency restriction does nothing. That is
precisely why the project starts at N=3: it isolates the game from the graph. The restriction
only begins to bite at N=4.

**What seating shuffles, and what it doesn't.** Position 0 is always position 0. What moves is
which *accuracy* is sitting there. This trips people up later, because it means "position 0
won 30% of the time" is a meaningless statistic while "the 0.4 player won 30% of the time" is
the one that matters.

**`rng` is an object, not a global setting.** Look at section 1.2: `seated_ranked` and
`seated_random` come from the *same* generator and are still different draws, because drawing
from it moves it on. The alternative, `np.random.seed(0)`, sets one hidden global that anything
in the program can silently consume.

**Try changing `N`.** Set it to 5 or 8 at the top and re-run. Watch the "reachable neighbours"
check flip from "not restricting anything" to "the graph is doing real work".

## What it deliberately doesn't do

No turns, no shooting, no rewards, no probability. This step is geometry.

**Next:** `step2-plain-game/` puts an actual game on this shape — random turn order, accuracy
rolls, elimination, winning — and ends by reproducing the exact survival probabilities
8/27, 9/27, 10/27.
