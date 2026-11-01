# The theory behind deliverable 03

This document explains why the mathematics in this folder is what it is. The ten steps show what was built and what it measured; none of them explains why a policy gradient has the shape it has, why the critic is given something the actor never sees, or why measuring who an agent shoots is not an optional extra.

## How to read this

Every section says the same thing twice. First in ordinary language, in a sentence you could say out loud. Then in the formal language, with every symbol named on first use, including summation indices.

The two registers are both necessary. The everyday sentence is what you understand with; the formal statement is what lets you have an accurate conversation with someone who already knows the field. The test of this document is not whether it can be read — it is whether, afterwards, you can use the formal terms correctly in conversation without having memorised them.

Each section ends by naming the function in this folder that implements it, as `file:line`. Anything not implemented here gets at most a sentence, in the last section. This is not a textbook.

## 1. The game as a formal object

**Everyday.** A game is four things: the situations you can be in, the moves you can make, a rule for what happens next, and a payoff at the end. Writing those four things down precisely is what lets you compute an answer instead of arguing about one.

**Formal.** This is a *stochastic game*, also called a Markov game, written

$$G = \langle \mathcal{N},\ \mathcal{S},\ \{\mathcal{A}_i\}_{i \in \mathcal{N}},\ P,\ \{r_i\}_{i \in \mathcal{N}},\ \gamma \rangle$$

where $\mathcal{N} = \{1, \dots, N\}$ is the set of players, $\mathcal{S}$ is the set of states, $\mathcal{A}_i$ is the set of actions available to player $i$, $P$ is the transition rule, $r_i$ is player $i$'s reward function, and $\gamma$ is the discount factor.

Filling those in for a gruel: a **state** is who is still alive together with the seating, so $\mathcal{S}$ is the set of non-empty subsets of the $N$ positions paired with an arrangement. The **action set** is the same three choices for every player, $\mathcal{A}_i = \{\text{shoot left},\ \text{shoot right},\ \text{abstain}\}$, so $|\mathcal{A}_i| = 3$; in code these are `SHOOT_LEFT`, `SHOOT_RIGHT` and `MISS`, and `N_ACTIONS` is 3. The **transition** $P(s' \mid s, a)$ is stochastic because a shot by a player with accuracy $p$ lands with probability $p$ and misses with probability $1 - p$; on a hit the circle contracts and the two neighbours of the dead player become neighbours of each other. The **discount** is $\gamma = 1$, which matters in section 2.

**Why this is a stochastic game and not a Markov decision process.** An MDP has exactly one decision maker, so the only source of uncertainty is the environment. Here several agents choose at once, and the best action for any one of them depends on what the others do. From one agent's point of view the other players are part of the environment — but they are *learning*, so that environment changes as training proceeds. Section 6 is about what that costs.

**It is also partially observed.** No agent sees $s$. Each sees an observation $o_i$ of exactly four numbers: its own accuracy, its left neighbour's accuracy, its right neighbour's accuracy, and the fraction of the field still alive. That makes this formally a *partially observable stochastic game*. Note what is deliberately absent: an agent is not told its own position, the full seating, or anything about players further round the circle, so it cannot tell where it sits in the global ranking. Four numbers whatever $N$ is — which section 7 shows is the reason one trained actor can play any player count.

**Turn order is a draw, not a rotation.** One player is chosen uniformly at random from those still alive, *with replacement*, so the same player can act twice in a row and one agent may make eight decisions in an episode while another makes one. This is the random-order variant of the game, and it is the single fact that most shapes the rest of this document.

**The horizon is not fixed.** An episode ends when one player remains, or when a stagnation limit of $10 \times n_{\text{alive}}$ consecutive misses is reached, or at a hard cap. So $T$, the length of an episode, is itself random.

In the code: `GruelEnv.observe` at `step9-player-count/gruel_env.py:216` returns the four numbers and its docstring states what is absent; `draw_actor` at `gruel_env.py:191` performs the uniform draw with replacement; `stagnation_threshold` at `gruel_env.py:212` scales the stuck-limit as players die.

## 2. What the agent maximises, and the hinge of the whole project

**Everyday.** The agent is trying to make its total future reward as big as possible. In this game, under the plain reward, "total future reward" and "did I win" are the same thing — which is what makes a trained agent's numbers comparable with an exact answer.

**Formal.** The *return* from time $t$ is the discounted sum of what arrives after it,

$$G_t = \sum_{k=0}^{T-t-1} \gamma^{k} \, r_{t+k+1}$$

where $k$ indexes steps forward from $t$, $r$ is the reward received at that step, $T$ is the (random) time the episode ends, and $\gamma \in [0, 1]$ discounts later rewards. Here $\gamma = 1$, so the return is a plain undiscounted sum.

The *value* of a state under policy $\pi$ is the expected return from it,

$$V^{\pi}(s) = \mathbb{E}_{\pi}\!\left[ G_t \mid s_t = s \right]$$

where $\mathbb{E}_{\pi}$ is the expectation over episodes played by $\pi$.

**Now the hinge.** Under the `terminal` reward profile the only non-zero entry is the win, worth $1.0$; being eliminated, killing someone and surviving a round are all worth exactly $0.0$. With $\gamma = 1$ and only one non-zero reward, which arrives at most once and only to the winner, the return collapses to an indicator:

$$G_t = \mathbb{1}\!\left[\text{this player wins the episode}\right]$$

and therefore

$$V^{\pi}(s) = \mathbb{E}_{\pi}\!\left[\mathbb{1}[\text{win}] \mid s\right] = \Pr\nolimits_{\pi}(\text{win} \mid s).$$

**One exception, and why it does not matter in practice.** An episode that stagnates or hits the hard cap has no winner, and `split_among_survivors` pays each player still standing $1/n_{\text{alive}}$. So strictly the return is $\mathbb{1}[\text{win}]$ plus that share in a stagnated episode, and $V^{\pi}(s)$ is the probability of winning plus the expected share. Every agent measured in this folder stagnates in 0.0000 of its episodes, so the extra term is zero to four decimals wherever a comparison is made. It would matter for a policy that held fire long enough to stall the game, which is why stagnation is measured on every run.

**A value estimate is, to that precision, an estimated survival probability.** That is not an analogy. It is why folder 02, which computes $\Pr(\text{win})$ exactly by backward induction, and folder 03, which trains an agent and measures how often it wins, are producing *the same quantity* and can be compared digit for digit. It is also why `terminal` is the only profile this folder gates against ground truth: section 9 shows what breaks otherwise.

In the code: `REWARD_PROFILES` at `step9-player-count/gruel_env.py:37` holds both profiles, and `terminal` is `{win: 1.0, eliminated: 0.0, kill: 0.0, survive: 0.0}`. The refusal that enforces this is `compare_to_exact` at `step9-player-count/step9_player_count.py:620`, which raises `WrongVariant` rather than return a comparison it should not make.

## 3. Policy gradient from first principles

**Everyday.** You cannot ask "what would have happened if I had done something else", because you only did one thing and the game moved on. So you do the only thing available: whatever you did, if the episode went well, make that more likely next time.

**Formal.** Write the policy as $\pi_{\theta}(a \mid s)$, a probability distribution over actions given a state, controlled by parameters $\theta$ — the network weights. The objective is the expected return,

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}}\!\left[ G_0 \right]$$

where $\tau$ is a whole episode, called a trajectory, drawn by playing $\pi_\theta$.

**The obstacle.** To improve $J$ by gradient ascent you need $\nabla_{\theta} J(\theta)$, the vector of partial derivatives of $J$ with respect to every parameter. But $\theta$ influences $J$ through *which actions were sampled*, and you cannot differentiate through a sampling operation — there is no derivative of "a 3 came up".

**The way round it** is the *score-function* identity. For any differentiable positive $f$, $\nabla_{\theta} \log f = \nabla_{\theta} f / f$, so $\nabla_\theta f = f \nabla_\theta \log f$. Applying that to the probability of a trajectory turns the derivative of an expectation into an expectation of a derivative:

$$\nabla_{\theta} J(\theta) = \mathbb{E}_{\pi_\theta}\!\left[ \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t) \, G_t \right]$$

Every symbol: $\nabla_\theta$ is the gradient with respect to the parameters; $\mathbb{E}_{\pi_\theta}$ is an average over episodes played by the *current* policy; $\log \pi_{\theta}(a_t \mid s_t)$ is the log-probability the policy gave to the action it actually took; and $G_t$ is that decision's return from section 2.

**In words, and this is the sentence to be able to say:** push up the log-probability of what you did, in proportion to how well it went.

The right-hand side is an average of things you can compute from played episodes, so it can be estimated by playing some and averaging. This estimator has several names and they mean the same thing: the *likelihood-ratio* estimator, the *score-function* estimator, and — as the algorithm built around it — *REINFORCE*.

Two consequences worth having ready. Because the expectation is over episodes played by the current policy, the data is stale the moment $\theta$ changes, which makes this *on-policy* and is the problem section 5 solves. And because $\log \pi \le 0$ always while $G_t \ge 0$ under `terminal`, the loss at this stage cannot go negative — a negative loss in step 1 would be a bug, not progress.

In the code: `episode_loss` at `step1-reinforce/step1_reinforce.py:234` computes $-\log \pi_\theta(a \mid s) \cdot G$ for each decision and averages, negated because optimisers minimise rather than maximise.

**What step 1 measures, which is the motive for section 4.** Under `terminal`, $G_t = 0$ for every decision made by every player who lost, so those terms vanish and only the winner's actions receive any gradient at all. Step 1 counts the discarded share over 200 untrained episodes and prints it: `0.280`. Below half, because the killing shot is always the winner's, and a player drawn more often is both likelier to win and to have made more decisions.

## 4. Why that is not enough: variance

**Everyday.** "Do more of what worked" judges every move in a lost game as worthless, including the good ones, and every move in a won game as valuable, including the careless ones. The signal is real but buried in noise. The fix is to compare what happened against what you had expected, rather than against zero.

**Formal.** Subtract a *baseline* $b(s)$ from the return:

$$\nabla_{\theta} J(\theta) = \mathbb{E}\!\left[ \nabla_{\theta} \log \pi_{\theta}(a \mid s) \left( G_t - b(s) \right) \right]$$

This is still the same gradient — the estimator remains **unbiased** — for *any* $b$ that does not depend on the action. The proof is three lines and is worth being able to reproduce, because it is exactly the question a supervisor asks next. The extra term that subtracting $b(s)$ introduces is

$$\mathbb{E}_{a \sim \pi_\theta}\!\left[ \nabla_{\theta} \log \pi_{\theta}(a \mid s) \right] b(s) = b(s) \sum_{a} \pi_{\theta}(a \mid s) \frac{\nabla_{\theta} \pi_{\theta}(a \mid s)}{\pi_{\theta}(a \mid s)} = b(s) \, \nabla_{\theta} \sum_{a} \pi_{\theta}(a \mid s) = b(s) \, \nabla_{\theta} 1 = 0$$

where the sum over $a$ runs across all three actions. The probabilities sum to $1$ for every $s$, a constant, whose gradient is zero. So the subtraction changes the variance of the estimate but not its mean — which is the entire reason it is allowed.

The most useful baseline is the state's own value, $b(s) = V(s)$. The result is the *advantage*,

$$A(s, a) = G_t - V(s)$$

read as: how much better this action turned out than the situation was worth. A good move in a lost game now scores positively, because it beat a low expectation.

$V$ is not known, so it is learned alongside the policy by a second network — the *critic* — trained to predict the return by squared error. The policy is then the *actor*. An actor plus a learned baseline is an *actor-critic* method.

One caution the code makes concrete: an *untrained* critic outputs arbitrary numbers, so the advantage signs it produces are arbitrary too. Step 2 measures this rather than assuming it, and finds no negative advantages at all before training, because the untrained critic's outputs happened to be negative. Being able to say why that is expected rather than broken is the point of measuring it.

In the code: the advantage is one line, `advantages = batch["returns"] - batch["values"]` at `step9-player-count/step9_player_count.py:385`. The line immediately after it standardises the advantages to zero mean and unit variance, which is a convenience for step-size stability rather than part of the theory — in expectation it rescales the gradient without turning it.

## 5. PPO: reusing a batch without falling off a cliff

**Everyday.** Collecting experience is the expensive part, so you would like to improve several times from one batch. But each improvement makes the batch describe a policy you no longer have, and past some point the data is actively misleading. PPO takes several steps from one batch and simply refuses to reward moving far.

**Formal.** The batch was played by the old parameters $\theta_{\text{old}}$. To estimate the objective for new parameters $\theta$ using that same data you must reweight each observed action by how much more or less likely the new policy makes it. That is *importance sampling*, and the weight is the *importance ratio*

$$r_t(\theta) = \frac{\pi_{\theta}(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$$

which equals $1$ for every sample before the first update, and drifts away from $1$ as $\theta$ moves. Computed in practice as $\exp(\log \pi_\theta - \log \pi_{\theta_{\text{old}}})$, because the log-probabilities are what the network gives and subtracting is better conditioned than dividing.

Importance sampling is unbiased but its variance grows without bound as the ratio departs from $1$, so the reuse has to be limited. The principled version constrains how far the policy may move in distribution — a *trust region* — for example by requiring $\mathrm{KL}\!\left(\pi_{\theta_{\text{old}}} \,\|\, \pi_{\theta}\right) \le \delta$, where $\mathrm{KL}$ is the Kullback–Leibler divergence, a measure of how different two distributions are. That is TRPO, and enforcing it needs a constrained optimisation at every step.

PPO gets most of the same effect for almost nothing, by clipping:

$$L^{\text{CLIP}}(\theta) = \mathbb{E}\!\left[ \min\!\left( r_t(\theta) A_t,\ \ \mathrm{clip}\!\left(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon\right) A_t \right) \right]$$

where $\mathrm{clip}$ holds its first argument inside the interval and $\epsilon$ is the clip width, $0.2$ here.

**The $\min$ is the whole trick, and it is asymmetric on purpose.** Take the *worse* of two quantities: the plain ratio-weighted advantage, and the same thing with the ratio held inside the interval. If moving further would look good, the clipped branch caps the gain, so there is no reward for moving far. If moving further looks bad, the unclipped branch keeps the full penalty, so there is always pressure back. The policy can be pulled back but not pushed away. Step 2 prints this as a table of the loss at ratios from $0.5$ to $1.5$ for advantage $+1$ and $-1$, which shows the asymmetry directly rather than describing it.

The full objective adds two more terms:

$$L(\theta, \phi) = L^{\text{policy}} + c_v \, L^{\text{value}} - c_e \, H\!\left[\pi_{\theta}\right]$$

where $\phi$ are the critic's parameters, $L^{\text{value}}$ is the squared error between the critic's prediction and the observed return, $H[\pi_\theta]$ is the *entropy* of the action distribution, and $c_v = 0.5$ and $c_e = 0.01$ weight them. Entropy is a measure of how spread out a distribution is; it is *subtracted* from a loss being minimised, so higher entropy is rewarded, which discourages the policy from collapsing onto one action before it has explored.

One diagnostic worth naming because it is how you tell whether any of this is doing work: the *clip fraction*, the share of ratios that fall outside the interval. Step 2 measures it, and the shape to want is a high fraction on the first update falling to a small mean — the clip doing real work while the policy moves fastest, then ceasing to matter. Never binding means the updates are too small to need it; binding constantly means the learning rate is too high.

The method is due to Schulman and co-authors, 2017.

In the code: `policy_ratio` at `step2-ppo/step2_ppo.py:277` and `clipped_objective` at `step2-ppo/step2_ppo.py:302` build it up in isolation; the assembled update, with all three terms and four epochs of shuffled minibatches over one batch, is `ppo_update` at `step9-player-count/step9_player_count.py:384`.

## 6. From one agent to many: MAPPO

**Everyday.** Single-agent learning assumes the world reacts the same way to the same move. With several learners that assumption is simply false, because the others are getting better at the same time as you.

**Formal.** From agent $i$'s viewpoint the transition and reward it experiences depend on the other agents' policies $\pi_{-i}$, the standard notation for "everyone except $i$". Those policies are themselves being updated, so the process agent $i$ faces is *non-stationary*: the target it is estimating moves while it estimates it. The convergence guarantees of single-agent policy gradient assume a fixed environment and do not carry over. This is the central difficulty of multi-agent reinforcement learning, and it is honest to say that MAPPO manages it rather than solves it.

Two structural choices manage it here.

**Self-play.** There is no fixed opponent. Every player is the same policy, so $\pi_{-i} = \pi_i$ and improving against the current opponent is improving against yourself. The opponent distribution still moves, but it moves *with* the learner rather than adversarially.

**Parameter sharing.** One network, many seats. A single `SharedActorCritic` is constructed per run and every player's action is drawn from it, so the players are not different agents — they are the same parameters acting from different seats on different observations. Three things follow. Every episode yields training data for one network rather than $N$ of them, so the sample cost does not multiply by the player count. There is no possibility of one seat's network being better trained than another's, which would confound the survival-by-rank measurement this folder exists to make. And because the observation already contains everything that distinguishes one player from another — its own accuracy and its two neighbours' — a separate network per seat would have to learn the same reasoning $N$ times and could not transfer to a different $N$.

MAPPO is then PPO plus a centralised critic plus parameter sharing, described by Yu and co-authors, 2022. The centralised critic is section 7.

In the code: `class SharedActorCritic` at `step9-player-count/step9_player_count.py:213`, instantiated once per run inside `train_one` at `step9_player_count.py:736`; the batch collected across all positions in `collect_batch` at `step9_player_count.py:278`, which keys its per-player lists by position and then flattens them into one batch.

## 7. The critic, and why it is $V(s, i)$ rather than $V(s)$

**Everyday.** The critic's job is to answer "how good is this situation". With more than one player that question is incomplete, because a situation good for one player is bad for another. So the critic has to be told *whose* prospects it is being asked about.

**Formal.** The critic is a function of two arguments,

$$V_{\phi}(s, i) \approx \Pr(\text{player } i \text{ wins} \mid s)$$

where $\phi$ are the critic's parameters and $i$ is the player being scored. Concretely its input is the concatenation of two vectors: `global_state` emits each position's accuracy together with a flag for whether that position is still alive, which is $2N$ numbers, and `one_hot` emits a vector of $N$ numbers that is zero everywhere except a $1$ at the position being scored. The total input width is $3N$ — so $9$, $15$ and $21$ at $N = 3$, $5$ and $7$.

**Why the one-hot is load-bearing, and this is the part most worth being able to defend.** Recall from section 1 that the actor is drawn at random with replacement. A player can therefore win having never fired a shot, because everyone else shot each other while it was never selected; and under `terminal` that player's return is $1.0$, identical to the return of a player who won by shooting well. The return is *inherited* rather than earned. Now suppose the critic saw only $s$. The same state would appear in the batch labelled with different returns depending on which player happened to be scored, and the best the critic could do is predict some average across players. The advantage $A = G - V(s)$ would then be dominated by *who* the decision belonged to rather than by whether the decision was good, which is precisely the noise the baseline was introduced to remove. Adding the one-hot makes the target well defined: each row asks about a specific player, and the critic can learn that the strongest seat's prospects differ from the weakest's.

**This asymmetry is the architecture, not an accident.** The actor's input is `OBSERVATION_SIZE`, which is $4$, whatever $N$ is. The critic's input is $3N$, which grows with $N$. So:

| | input | scales with $N$? | available at execution time? |
|---|---|---|---|
| actor | $4$ | no | yes |
| critic | $3N$ | yes | no |

The same trained actor can be dropped into a game with a different player count, because nothing in its input mentions $N$. The critic cannot, and does not need to: it is **discarded after training**. It exists only to reduce the variance of the actor's gradient, and at deployment the actor is all that runs.

That pattern has a name — *centralised training, decentralised execution*, or CTDE. During training you may use anything you can compute, including things no player could know. At execution each agent must act on its own observation alone.

**The critic's sight of the global state is a training-time convenience, not a channel between players.** Nothing an agent conditions on when acting comes from anywhere but its own four numbers.

Step 3 demonstrates the asymmetry instead of asserting it, by taking a network trained at $N = 3$ and offering each head an $N = 5$ input. The actor accepts the four-number observation unchanged and returns three logits. The critic, handed the corresponding $N = 5$ global state, cannot: its input width is $9$ at $N = 3$ while $N = 5$ needs $15$, so the multiplication is undefined and it raises a shape error, which the notebook catches and prints.

In the code: `global_state` at `step9-player-count/step9_player_count.py:233`, `one_hot` at `step9_player_count.py:243`, and the two heads `action_logits` and `value` on `SharedActorCritic` at `step9_player_count.py:224` and `step9_player_count.py:228`. The critic's width is set once, as `critic_input_size = 2 * n_players + n_players`, and the step prints the resulting widths for all three player counts rather than claiming them.

**One further detail that belongs here because it is easy to get wrong.** The return each decision is trained against is the *reward-to-go*: what the player ended the episode with, minus what it had already banked at the moment it decided. With no discounting that is exactly the undiscounted sum of what arrived afterwards. It is read from the environment's own arithmetic, `game.episode_returns`, rather than assumed. Under `terminal` this reduces exactly to "$1$ for the winner, $0$ for everyone else", which is why it is invisible until section 9's reward changes; the notebook asserts that reduction rather than trusting it. `collect_batch` at `step9_player_count.py:278`, with the subtraction at `step9_player_count.py:346`.

**Measured rather than argued.** The case above says a critic seeing one player's four numbers is at a disadvantage. Step 10 measures how large that disadvantage is by training the same agent with exactly such a critic, which is independent PPO with parameter sharing. At $N=3$ and $N=5$ the local critic passes the same gates; at $N=7$, where both fail, the centralised critic's worst-rank ratio is 0.149 higher, paired over five seeds, and its targeting is closer to the equilibrium's. Its value estimates are not better on average, with slightly lower explained variance at every size. Folder 04 step 8 repeats the comparison at $N=10$ and $N=15$, where there is no exact answer, using the switch test: neither critic is measurably closer to an equilibrium, and the local critic again predicts the return slightly better. So in this project the centralised critic is an advantage at $N=7$ only, not a requirement for learning the game.

## 8. Equilibrium: the vocabulary, and where the answer comes from

The exact answers this folder is measured against are computed in **[folder 02](../02-exact-solver/README.md)**, and the reasoning is set out there rather than repeated here. What follows is only the vocabulary needed to talk about what 03 is comparing itself to.

A **Nash equilibrium** is a profile of strategies, one per player, in which no player can do better by changing its own strategy while the others keep theirs. It is a statement about unilateral deviation only; it says nothing about what players could achieve by coordinating.

A **Markov** or **stationary** strategy depends only on the current state, not on the history of how the game arrived there. Every strategy considered in this project is Markov, and the agent's architecture enforces it: the actor sees four numbers about the present and has no memory.

A **Markov perfect equilibrium** is a profile that is a Nash equilibrium *in every state*, not merely from the opening position. This is the stronger and more useful property, because it rules out threats that would never be carried out.

The **one-shot deviation principle** is the tool that makes this computable. If no player gains by changing a single action at a single state, holding everything else fixed, the profile is an equilibrium. Folder 02's step 5 establishes why that test is *sufficient* here, which is the part worth being able to explain out loud: a kill strictly shrinks the set of living players, so the graph of states is acyclic and there is no way to return to a state and coordinate a plan across two visits; and a miss leaves the state unchanged, so the recursion absorbs that self-loop by conditioning on which shot lands next, normalising the hazards and expressing the value as a weighted average over *successor* states rather than over itself. Together these mean that passing the one-shot test gives Markov perfection, not merely Nash play from the start.

**Backward induction** is then the method: solve the states with fewest players first, where the answer is immediate, and work upward, since every state's value depends only on states with strictly fewer players alive.

**Where it stops, and why that is the argument for this folder existing.** Folder 02 solves $N = 3$ through $N = 8$ and is *refused* at $N = 9$: in one state, none of the available pure combinations is stable, so there is no pure equilibrium to report. The limit is **not cost**: the solver would have been fast enough to go further. That state does have a *mixed* equilibrium, in which two players randomise between a shot and holding fire, so the limit is on backward induction over pure strategies rather than on the game. A solver over mixed strategies could continue, and this project does not build one. Above the solver's reach a trained agent is not being checked against a known answer, so what it does is described rather than graded.

The three exact profiles this folder gates against are $N = 3$, $N = 5$ and $N = 7$, and they are three genuinely different shapes rather than three instances of one trend, which is what makes training at three counts worth the compute. At $N = 3$ the ordering is monotone and the weakest player is best off, at $0.37037$ against $0.33333$ and $0.29630$. At $N = 5$ it is not monotone. At $N = 7$ the best-off player is not the weakest at all. Each step states its own profile and compares against it; step 9 does all three.

## 9. Why reward shaping is a separate question from the game

**Everyday.** Changing the reward changes what the agent is *told*, not what the game *does*. The rules, the accuracies and who wins are all untouched. That sounds obvious and has a consequence that is not: it means any difference in behaviour between two reward profiles arrived through learning and nothing else.

**Formal.** The `shaped` profile keeps the win at $1.0$ and adds three hints: $-1.0$ for being eliminated, $+0.1$ for a kill, and $+0.001$ for surviving a round. Under it the return is no longer an indicator, so section 2's collapse fails, $V^\pi$ is no longer a win probability, and the exact answers are **not** the right comparison — the agent is maximising a different objective from the one the solver maximises. Judging it against the solver's numbers would be marking it on a question it was not asked.

**What shaping does and does not change.** It does not invert which outcome pays best. Net of the certain $-1.0$ for dying, a prolific killer who dies still trails a winner. What it changes is the *gradient*: a kill is a certain, immediate $+0.1$, whereas the win is distant and uncertain, so the shaped agent is pulled toward a nearer signal. Whether that helps or misleads is an empirical question, which is why the reward profile is a dial in this folder rather than a setting chosen once.

There is a formal result worth knowing here, because it says why the honest treatment is to measure. Ng, Harada and Russell (1999) showed that a shaping term of the *potential-based* form $\gamma \Phi(s') - \Phi(s)$, for any function $\Phi$ of the state, leaves the set of optimal policies unchanged. This project's shaping is **not** of that form — a kill bonus is not the difference of a potential — so no invariance guarantee applies and the shaped objective genuinely may have different optima. That is the precise reason it cannot be assumed harmless.

Step 8 also gives the size of the hint, which is what decides whether it matters: the total kill bonus available in an episode, as a fraction of the win reward, is $0.10$ at $N = 3$, $0.30$ at $N = 5$, $0.80$ at $N = 10$ and $1.30$ at $N = 15$. At $N = 5$ the hint is small, and step 8 measures no difference between the two profiles that survives its error bars. At $N = 15$ the same mechanism is large. A null result at one scale and a large effect at another are the same mechanism seen twice, not a contradiction.

In the code: `REWARD_PROFILES` at `step9-player-count/gruel_env.py:37`; the refusal `compare_to_exact` at `step8-reward-profile/step8_reward_profile.py:691`, keyed on the reward name, and its counterpart at `step9_player_count.py:620`, which checks the reward, the seating and whether an exact profile is carried for that $N$.

## 10. Identification: why counting wins is not enough

**Everyday.** Two players can win the same number of games while playing in completely different ways. So the win count, however carefully measured, cannot tell you *how* someone plays. If the question is what an agent learned, wins are the wrong instrument — not an imprecise one, the wrong one.

**Formal.** This is an *identification* problem, and stating it that way is what makes it precise. Let $\mu(\pi)$ be the survival profile induced by policy $\pi$ — the vector whose entries are the win probability of each accuracy rank. The map $\mu$ is **not injective**: there exist distinct policies $\pi \ne \pi'$ with $\mu(\pi) = \mu(\pi')$. Consequently $\mu$ does not identify $\pi$, and no estimator built only on outcomes can distinguish those policies *however many episodes are played*. This is a limitation of the statistic, not of the sample size, which is why more compute does not fix it and a second statistic does.

The concrete case that matters here: a flat survival profile is produced both by a near-uniform equilibrium and by a policy that discriminates on nothing at all. Those are opposite findings — one says the game is nearly fair, the other says the agent learned nothing — and survival numbers cannot separate them.

**The second statistic is targeting.** Given that a player fired, and had two distinct neighbours of different strength, how often did it shoot the stronger one. Being able to state its three exclusion rules is the test of whether the measurement is understood, because each one removes a case where the question has no answer:

1. **It must have fired.** A player that abstained selected no target, so it contributes no evidence about targeting.
2. **More than two players must be alive.** At two, the left and right neighbours are the same player, so there is no choice to observe.
3. **The two neighbours must differ in accuracy.** If they are equally strong there is no stronger one to choose.

And the refusal that makes it honest: when no decisions are eligible the rate is `None`, not $0.0$. Returning $0.0$ would assert that the player always shot the *weaker* neighbour, which is the opposite of "it never chose". The all-abstain baseline is the case that proves the point, and it is measured rather than argued.

**The three signatures**, which is what targeting buys:

| signature | at the extremes | in the middle | dependence on survivors |
|---|---|---|---|
| matched the equilibrium | $\approx 1.0$ | $\approx 0.3$ | varies strongly |
| imitated `shoot_strongest` | high | high | flat |
| discriminated on nothing | $\approx 0.5$ | $\approx 0.5$ | none |

The middle column is the discriminating one. The exact equilibrium has only the two extreme players firing at their stronger neighbour, while the middle players mostly fire at their *weaker* one. Neither `shoot_strongest` nor a coin flip produces that shape, but one fixed rule does: under ranked seating the circle stays sorted, so always shooting the left-hand neighbour (Archetti's rule restricted to the neighbours) has the same U-shape, and only the size of the middle values separates it from the equilibrium (folder 04 step 3). So a policy that looks like "shoot the strongest" everywhere has not found the equilibrium, however respectable its survival numbers.

**Conditioning on the number of survivors is not optional.** Folder 02 computes the equilibrium's targeting at $N = 5$ as $0.40$ with five alive, $0.50$ with four, and $1.00$ with three. A single pooled average over those describes a game nobody plays. Step 5 makes this concrete with `split_by_seat`, a policy constructed so that its average misdescribes every player in it, and calibrates the survivor-count axis against `switch_at_three`, a policy built to have a known dependence, recovering the constructed step exactly.

**Two cautions about $N = 3$, both properties of the game rather than of any agent.** At $N = 3$ the equilibrium *is* "shoot the stronger neighbour", so flat targeting near $1.0$ is the correct answer there and looking for a U-shape is the wrong test. And at $N = 3$ the survivor-count axis is empty: play starts with three alive, the first kill leaves two, and two-alive decisions are excluded by rule 2 — so every eligible decision has exactly three alive, and the table has one row. A one-row table showing no variation looks exactly like the "discriminated on nothing" signature and is nothing of the kind, which is why the instrument is calibrated at $N = 5$.

In the code: `eligible` at `step9-player-count/step9_player_count.py:515` holds the three rules in the order given above, `chose_stronger` at `step9_player_count.py:524` answers the question itself, `targeting_rate` at `step9_player_count.py:533` returns `None` when there are no decisions, and `targeting_by_alive` at `step9_player_count.py:549` conditions on the survivor count. The signature comparison is `signature_of` at `step5-targeting/step5_targeting.py:847`.

## The hyperparameters, and which are theory and which are taste

Only some of these carry meaning. The clip width $\epsilon$, the value and entropy coefficients and the number of epochs are part of the method described above. The rest are budget.

| symbol | value | what it is |
|---|---|---|
| $\epsilon$ | `CLIP = 0.2` | clip width, section 5 |
| $c_v$ | `VALUE_COEFFICIENT = 0.5` | weight on the critic's squared error |
| $c_e$ | `ENTROPY_COEFFICIENT = 0.01` | weight on the entropy bonus |
| $\gamma$ | $1$, by omission | no discounting, section 2 |
| — | `EPOCHS = 4` | passes over each batch, section 5 |
| — | `LEARNING_RATE = 0.003` | Adam step size |
| — | `HIDDEN_SIZE = 64` | one hidden layer, $\tanh$ activation |
| — | `UPDATES = 2000`, `EPISODES_PER_UPDATE = 96` | budget: $192{,}000$ training episodes |
| — | `MINIBATCH_SIZE = 256` | minibatch within an epoch |
| — | `EVALUATION_EPISODES = 10000`, `SEEDS = [0, 1, 2]` | measurement, with exploration still on |
| — | `GATE_THRESHOLD = 0.95` | the pass mark against the exact profile |

The network is deliberately small: one hidden layer of $64$ units with a $\tanh$ activation, for both heads. A circle of at most fifteen players observed through four numbers does not need depth, and a small network makes the training cheap enough to run several seeds — which matters more, because a single seed cannot show whether a result is reproducible.

## What this document deliberately does not cover

Each of these is real and none of it is implemented in this folder, so each gets a sentence and no more.

**Generalised advantage estimation (GAE)** interpolates between the high-variance single-sample return and the low-variance but biased one-step bootstrap, using a parameter $\lambda$; here the episodes are short — a handful of decisions per player — so the plain reward-to-go is used and $\lambda$ never appears.

**Discounting** with $\gamma < 1$ trades away some of the future to reduce variance and to keep infinite-horizon returns finite; with $\gamma = 1$ the collapse in section 2 is exact, which is worth more here than the variance reduction.

**Recurrence and memory** would let an agent condition on the history rather than only the present state, which matters when the observation hides something that past observations revealed; every strategy here is Markov by construction and the actor has no memory.

**Off-policy methods**, such as Q-learning and its deep variants, learn from data collected by a different policy and so can reuse a replay buffer indefinitely; PPO is on-policy, which is why section 5 needs the importance ratio at all.

## Where to go next

The mathematics above is exercised in order by the ten steps, and each step's README states what number to expect before showing it.

- Sections 3 and 4 are built in [step 1](step1-reinforce/README.md) and [step 2](step2-ppo/README.md), at $N = 2$ where the answer is a single fraction.
- Sections 6 and 7 arrive together in [step 3](step3-mappo/README.md), which is where a third player makes "is this state good?" ambiguous.
- Sections 8 and 10 are the instruments, built in [step 4](step4-measurement/README.md) and [step 5](step5-targeting/README.md) and first used to pass or fail in [step 6](step6-n5-gate/README.md).
- Section 9 is the subject of [step 8](step8-reward-profile/README.md).
- [Step 9](step9-player-count/README.md) turns the player-count dial and carries the fullest description of the architecture as built, in its section *The agent, in full*.
