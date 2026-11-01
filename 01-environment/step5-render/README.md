# Step 5 — Drawing the game

## What this step is

Everything so far has been numbers. This step draws the game, and the reason is not decoration:
**contraction is very hard to believe from a table and obvious the moment you see it.**

There is one design decision here that matters more than all the drawing code put together.

When a player dies, you could re-space the survivors evenly around the circle. It would look
tidy. It would also make contraction **invisible** — the picture would always be a neat circle
and you would never see the new edge appear.

So instead: **every player keeps the angle they started at, forever.** Dead players stay in
their seats, drawn struck through. That leaves gaps, and the edge created by contraction visibly
jumps across the gap. It is drawn **dashed** so it is distinguishable from an original edge.

The one mechanism that makes this game interesting becomes the most obvious thing on screen.

## How to run it

Two things live here. The notebook draws still frames inline:

```bash
../.venv/bin/jupyter lab step5_render_frames.ipynb
```

About **10 seconds**. Produces three figures.

`play.py` is a live pygame window, and is a script rather than a notebook because a window
cannot live inside a notebook:

```bash
../.venv/bin/python play.py --n 6
../.venv/bin/python play.py --n 15 --policy random --fps 6
../.venv/bin/python play.py --n 5 --step
```

| Flag | Default | Meaning |
|---|---|---|
| `--n` | 6 | number of players |
| `--seed` | 0 | random seed |
| `--policy` | `shoot_strongest` | also `shoot_weakest`, `random`, `always_miss` |
| `--seating` | `random` | or `ranked` |
| `--fps` | 2.0 | frames per second |
| `--episodes` | 1 | how many to play in a row |
| `--step` | off | pause on every frame |

**`--step` is the one worth using.** At full speed a contraction flashes past. Stepping lets you
watch the dashed edge appear. `SPACE` advances one frame, `P` plays on without pausing, `Q` or
`Esc` quits.

## What you should see

**From the notebook.** The screen coordinates, position 0 at the top:

```
position 0: x=+0.000, y=+1.000
position 1: x=+0.866, y=+0.500
position 2: x=+0.866, y=-0.500
position 3: x=+0.000, y=-1.000
```

An episode replayed as a grid of frames, then the clearest single picture — the same 6-player
circle before and after one death:

```
p1's neighbours now: (0, 3)
p3's neighbours now: (1, 4)

p1 and p3 could not shoot each other before. Now they can.
```

In that figure the left panel is a plain grey hexagon. The right panel has p2 faded and struck
through in its seat, and a **dashed orange line** running from p1 across the gap to p3.

Then the N=15 legibility check, with 9 players dead:

```
alive: [0, 2, 5, 6, 9, 11]
edges now: [(0, 2), (2, 5), (5, 6), (6, 9), (9, 11), (11, 0)]
```

drawn as a ring of nine struck-through grey circles with dashed orange edges spanning the gaps
between the six survivors.

Seeding:

```
same seed gives an identical episode: True
```

**From `play.py`.** A 1000×720 window, one line of plain English at the top describing what just
happened, a status line, and a legend in the bottom-left saying what each mark *means* rather
than naming it. On exit it prints a summary per episode:

```
episode 0: ended sole_survivor after 12 steps, survivors [3], winner accuracy [0.72]
```

## What to look at

**Count the dashed edges as the episode goes on.** Every one is a pair of players who could not
reach each other when the episode started. By the time three of six are dead, most of the graph
is dashed — the game being played at the end is a different graph from the one it started on.

**The N=15 figure answers a real question.** The project needs to run at N=15, and a picture that
turns to spaghetti is useless. It stays readable, so no player-count limit is needed for
N=3 through N=15. A non-circle topology would need a proper graph layout algorithm, but the project
uses only the circle.

**Three kinds of line, and they must be told apart at a glance.** Solid pale grey is an original
edge; dashed orange is a contracted one; an arrow is a shot just fired, red if it hit and dark
grey if it missed. The colours are far apart on purpose: two greys 30 shades apart for live edges
and missed shots would be fine in a still frame and unreadable at 4 frames per second.

**Watch for this.** The first version used the angle $-\pi/2$ for position 0, which puts it at
the **bottom**, because matplotlib measures angles from the positive x-axis with y pointing up.
The correct formula is $\theta_i = \pi/2 - 2\pi i / n$ — the $\pi/2$ starts at the top, and
*subtracting* makes it go clockwise. I found it by looking at the rendered figure rather than by
reading the code.

**A second one.** Node labels originally showed one decimal place, so at N=6 the accuracies 0.56
and 0.64 both printed as "0.6" and two different players looked identical. Two decimals now.

**Try `--policy always_miss`.** Nobody ever dies, no edge is ever contracted, and after 60 steps
the game gives up and declares itself stuck. It is the clearest possible picture of the
"non-shooting" outcome the project has to be able to detect.

## What it deliberately doesn't do

**No GIF export.** The notebook shows frames inline and `play.py` shows a live window, which covers
reading and watching.

**No human action choice.** `play.py` lets you control the *pace* with `--step`, but the actions
always come from the policy. A keyboard-controlled mode would add nothing: the research question is
what agents learn, not what a human plays.

**No non-circle layouts.** Every topology here is a circle, so fixed angles are enough. Scale-free
or random graphs would need `networkx.spring_layout`.

**Next:** that completes deliverable 01. Go to `../../02-exact-solver/`, which stops measuring
survival with 100,000 random episodes and computes it exactly instead.
