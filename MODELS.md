# Trained models

Generated from the records in each step's `models/` folder.

Every model a step trains for its results is saved as a PyTorch `state_dict` (`<run>.pt`) with a record of the model (`<run>.json`): its layers and parameter counts, what the actor and the critic see, every training setting and seed, the game, the software versions, the time it took, and hashes of the weights and of the code that produced it. Every model listed here passed four checks: its weights file and record are both present, the file is unchanged since it was saved, the loaded weights hash to the recorded value, and every layer's shape and parameter count matches the record.

**Two hashes.** `weights_sha256` is computed from the weight values, so the same model has the same hash wherever it was trained. `file_sha256` is the hash of the `.pt` file, which PyTorch does not write byte-for-byte the same way twice, so it only shows the file is unchanged.

## How to load one

Copy the network class from the notebook that trained the model (the record's `notebook` field), build it with the record's settings, and load the weights:

```python
import json, torch
card = json.load(open("models/<run>.json"))
model = SharedActorCritic(n_players=card["game"]["n_players"])            # steps without a critic_view argument
# model = SharedActorCritic(n_players=..., critic_view="local")            # 03 step 10 and 04 step 8
model.load_state_dict(torch.load("models/<run>.pt", weights_only=True))
model.eval()
```

A loaded model reproduces its step's measurements when it is measured with the same seeds, because every measurement in this repository resets its own random streams.

## Summary

| Step | Models | Model class | Algorithm | Parameters (actor + critic) |
|---|---|---|---|---|
| `03-agent-architecture/step1-reinforce` | 1 | PolicyNetwork | REINFORCE, one shared policy, no critic | 515 + 0 |
| `03-agent-architecture/step10-critic-ablation` | 30 | SharedActorCritic | PPO with a shared actor and a centralised critic; PPO with a shared actor and a local critic | 515 + 1089, 515 + 1473, 515 + 385, 515 + 705 |
| `03-agent-architecture/step2-ppo` | 34 | ActorCritic, PolicyNetwork | PPO, one shared actor and a critic that sees the actor's four numbers; REINFORCE, one shared policy, no critic | 515 + 0, 515 + 385 |
| `03-agent-architecture/step3-mappo` | 1 | SharedActorCritic | PPO with a shared actor and a centralised critic | 515 + 705 |
| `03-agent-architecture/step5-targeting` | 1 | SharedActorCritic | PPO with a shared actor and a centralised critic | 515 + 705 |
| `03-agent-architecture/step6-n5-gate` | 5 | SharedActorCritic | PPO with a shared actor and a centralised critic | 515 + 1089 |
| `03-agent-architecture/step7-seating` | 6 | SharedActorCritic | PPO with a shared actor and a centralised critic | 515 + 1089 |
| `03-agent-architecture/step8-reward-profile` | 6 | SharedActorCritic | PPO with a shared actor and a centralised critic | 515 + 1089 |
| `03-agent-architecture/step9-player-count` | 10 | SharedActorCritic | PPO with a shared actor and a centralised critic | 515 + 1089, 515 + 1473, 515 + 705 |
| `04-scaling/step1-train-any-n` | 15 | SharedActorCritic | PPO with a shared actor and a centralised critic | 515 + 1089, 515 + 1473, 515 + 2049, 515 + 3009, 515 + 705 |
| `04-scaling/step2-switch-test` | 9 | SharedActorCritic | PPO with a shared actor and a centralised critic | 515 + 1089, 515 + 2049, 515 + 3009 |
| `04-scaling/step3-targeting` | 9 | SharedActorCritic | PPO with a shared actor and a centralised critic | 515 + 1089, 515 + 2049, 515 + 3009 |
| `04-scaling/step4-continuity` | 15 | SharedActorCritic | PPO with a shared actor and a centralised critic | 515 + 1089, 515 + 1473, 515 + 2049, 515 + 3009, 515 + 705 |
| `04-scaling/step5-reward-profile` | 12 | SharedActorCritic | PPO with a shared actor and a centralised critic | 515 + 2049, 515 + 3009 |
| `04-scaling/step6-seating` | 12 | SharedActorCritic | PPO with a shared actor and a centralised critic | 515 + 2049, 515 + 3009 |
| `04-scaling/step7-holding-fire` | 10 | SharedActorCritic | PPO with a shared actor and a centralised critic | 515 + 3009 |
| `04-scaling/step8-critic-ablation` | 20 | SharedActorCritic | PPO with a shared actor and a centralised critic; PPO with a shared actor and a local critic | 515 + 2049, 515 + 3009, 515 + 385 |
| `04-scaling/step9-direction` | 60 | SharedActorCritic | PPO with a shared actor and a centralised critic | 515 + 1473, 515 + 2049, 515 + 3009 |
| **all** | **256** | | | |

## The same configuration trained in more than one step

Steps copy their code forward rather than importing it, so several steps train the same configuration with the same seeds. If the copies are faithful, those runs produce the same model, and their `weights_sha256` agree. This is a check on the whole training pipeline across steps, not only on the saved files.

| Run | Steps | Identical weights |
|---|---|---|
| `n10_ranked_terminal_centralised_u2000_seed0` | 04-scaling/step1-train-any-n, 04-scaling/step2-switch-test, 04-scaling/step3-targeting, 04-scaling/step4-continuity, 04-scaling/step5-reward-profile, 04-scaling/step6-seating, 04-scaling/step8-critic-ablation, 04-scaling/step9-direction | yes |
| `n10_ranked_terminal_centralised_u2000_seed1` | 04-scaling/step1-train-any-n, 04-scaling/step2-switch-test, 04-scaling/step3-targeting, 04-scaling/step4-continuity, 04-scaling/step5-reward-profile, 04-scaling/step6-seating, 04-scaling/step8-critic-ablation, 04-scaling/step9-direction | yes |
| `n10_ranked_terminal_centralised_u2000_seed2` | 04-scaling/step1-train-any-n, 04-scaling/step2-switch-test, 04-scaling/step3-targeting, 04-scaling/step4-continuity, 04-scaling/step5-reward-profile, 04-scaling/step6-seating, 04-scaling/step8-critic-ablation, 04-scaling/step9-direction | yes |
| `n10_ranked_terminal_centralised_u2000_seed3` | 04-scaling/step8-critic-ablation, 04-scaling/step9-direction | yes |
| `n10_ranked_terminal_centralised_u2000_seed4` | 04-scaling/step8-critic-ablation, 04-scaling/step9-direction | yes |
| `n15_ranked_terminal_centralised_u2000_seed0` | 04-scaling/step1-train-any-n, 04-scaling/step2-switch-test, 04-scaling/step3-targeting, 04-scaling/step4-continuity, 04-scaling/step5-reward-profile, 04-scaling/step6-seating, 04-scaling/step7-holding-fire, 04-scaling/step8-critic-ablation, 04-scaling/step9-direction | yes |
| `n15_ranked_terminal_centralised_u2000_seed1` | 04-scaling/step1-train-any-n, 04-scaling/step2-switch-test, 04-scaling/step3-targeting, 04-scaling/step4-continuity, 04-scaling/step5-reward-profile, 04-scaling/step6-seating, 04-scaling/step7-holding-fire, 04-scaling/step8-critic-ablation, 04-scaling/step9-direction | yes |
| `n15_ranked_terminal_centralised_u2000_seed2` | 04-scaling/step1-train-any-n, 04-scaling/step2-switch-test, 04-scaling/step3-targeting, 04-scaling/step4-continuity, 04-scaling/step5-reward-profile, 04-scaling/step6-seating, 04-scaling/step7-holding-fire, 04-scaling/step8-critic-ablation, 04-scaling/step9-direction | yes |
| `n15_ranked_terminal_centralised_u2000_seed3` | 04-scaling/step7-holding-fire, 04-scaling/step8-critic-ablation, 04-scaling/step9-direction | yes |
| `n15_ranked_terminal_centralised_u2000_seed4` | 04-scaling/step7-holding-fire, 04-scaling/step8-critic-ablation, 04-scaling/step9-direction | yes |
| `n15_ranked_terminal_centralised_u2000_seed5` | 04-scaling/step7-holding-fire, 04-scaling/step9-direction | yes |
| `n15_ranked_terminal_centralised_u2000_seed6` | 04-scaling/step7-holding-fire, 04-scaling/step9-direction | yes |
| `n15_ranked_terminal_centralised_u2000_seed7` | 04-scaling/step7-holding-fire, 04-scaling/step9-direction | yes |
| `n15_ranked_terminal_centralised_u2000_seed8` | 04-scaling/step7-holding-fire, 04-scaling/step9-direction | yes |
| `n15_ranked_terminal_centralised_u2000_seed9` | 04-scaling/step7-holding-fire, 04-scaling/step9-direction | yes |
| `n3_ranked_terminal_centralised_u2000_seed0` | 03-agent-architecture/step10-critic-ablation, 03-agent-architecture/step9-player-count, 04-scaling/step1-train-any-n, 04-scaling/step4-continuity | yes |
| `n3_ranked_terminal_centralised_u2000_seed1` | 03-agent-architecture/step10-critic-ablation, 03-agent-architecture/step9-player-count, 04-scaling/step1-train-any-n, 04-scaling/step4-continuity | yes |
| `n3_ranked_terminal_centralised_u2000_seed2` | 03-agent-architecture/step10-critic-ablation, 03-agent-architecture/step9-player-count, 04-scaling/step1-train-any-n, 04-scaling/step4-continuity | yes |
| `n3_ranked_terminal_centralised_u400_seed0` | 03-agent-architecture/step3-mappo, 03-agent-architecture/step5-targeting | yes |
| `n5_ranked_terminal_centralised_u2000_seed0` | 03-agent-architecture/step10-critic-ablation, 03-agent-architecture/step6-n5-gate, 03-agent-architecture/step7-seating, 03-agent-architecture/step8-reward-profile, 03-agent-architecture/step9-player-count, 04-scaling/step1-train-any-n, 04-scaling/step2-switch-test, 04-scaling/step3-targeting, 04-scaling/step4-continuity | yes |
| `n5_ranked_terminal_centralised_u2000_seed1` | 03-agent-architecture/step10-critic-ablation, 03-agent-architecture/step6-n5-gate, 03-agent-architecture/step7-seating, 03-agent-architecture/step8-reward-profile, 03-agent-architecture/step9-player-count, 04-scaling/step1-train-any-n, 04-scaling/step2-switch-test, 04-scaling/step3-targeting, 04-scaling/step4-continuity | yes |
| `n5_ranked_terminal_centralised_u2000_seed2` | 03-agent-architecture/step10-critic-ablation, 03-agent-architecture/step6-n5-gate, 03-agent-architecture/step7-seating, 03-agent-architecture/step8-reward-profile, 03-agent-architecture/step9-player-count, 04-scaling/step1-train-any-n, 04-scaling/step2-switch-test, 04-scaling/step3-targeting, 04-scaling/step4-continuity | yes |
| `n5_ranked_terminal_centralised_u2000_seed3` | 03-agent-architecture/step10-critic-ablation, 03-agent-architecture/step6-n5-gate | yes |
| `n5_ranked_terminal_centralised_u2000_seed4` | 03-agent-architecture/step10-critic-ablation, 03-agent-architecture/step6-n5-gate | yes |
| `n7_ranked_terminal_centralised_u2000_seed0` | 03-agent-architecture/step10-critic-ablation, 03-agent-architecture/step9-player-count, 04-scaling/step1-train-any-n, 04-scaling/step4-continuity, 04-scaling/step9-direction | yes |
| `n7_ranked_terminal_centralised_u2000_seed1` | 03-agent-architecture/step10-critic-ablation, 03-agent-architecture/step9-player-count, 04-scaling/step1-train-any-n, 04-scaling/step4-continuity, 04-scaling/step9-direction | yes |
| `n7_ranked_terminal_centralised_u2000_seed2` | 03-agent-architecture/step10-critic-ablation, 03-agent-architecture/step9-player-count, 04-scaling/step1-train-any-n, 04-scaling/step4-continuity, 04-scaling/step9-direction | yes |
| `n7_ranked_terminal_centralised_u2000_seed3` | 03-agent-architecture/step10-critic-ablation, 04-scaling/step9-direction | yes |
| `n7_ranked_terminal_centralised_u2000_seed4` | 03-agent-architecture/step10-critic-ablation, 04-scaling/step9-direction | yes |

29 of 29 shared configurations have identical weights in every step that trained them.

## Every model

### `03-agent-architecture/step1-reinforce`

| Run | N | Seating | Reward | Critic input | Training episodes | Seed | Parameters | Seconds | weights_sha256 |
|---|---|---|---|---|---|---|---|---|---|
| `n2_ranked_terminal_reinforce_e6000_seed0` | 2 | ranked | terminal | none | 6000 | 2000 | 515 | 3.3 | `e37c270dd7b2` |

### `03-agent-architecture/step10-critic-ablation`

| Run | N | Seating | Reward | Critic input | Training episodes | Seed | Parameters | Seconds | weights_sha256 |
|---|---|---|---|---|---|---|---|---|---|
| `n3_ranked_terminal_centralised_u2000_seed0` | 3 | ranked | terminal | centralised, 9 inputs | 192000 | 0 | 1220 | 65.2 | `f29ae0f9b5ff` |
| `n3_ranked_terminal_centralised_u2000_seed1` | 3 | ranked | terminal | centralised, 9 inputs | 192000 | 1 | 1220 | 75.6 | `03fe4e8e9674` |
| `n3_ranked_terminal_centralised_u2000_seed2` | 3 | ranked | terminal | centralised, 9 inputs | 192000 | 2 | 1220 | 87.7 | `470df0cf9139` |
| `n3_ranked_terminal_centralised_u2000_seed3` | 3 | ranked | terminal | centralised, 9 inputs | 192000 | 3 | 1220 | 83.8 | `a3eb0fc38f4e` |
| `n3_ranked_terminal_centralised_u2000_seed4` | 3 | ranked | terminal | centralised, 9 inputs | 192000 | 4 | 1220 | 87.1 | `b90fa1ed6304` |
| `n3_ranked_terminal_local_u2000_seed0` | 3 | ranked | terminal | local, 4 inputs | 192000 | 0 | 900 | 85.9 | `d764a0a1bf0c` |
| `n3_ranked_terminal_local_u2000_seed1` | 3 | ranked | terminal | local, 4 inputs | 192000 | 1 | 900 | 87.2 | `692158d1a319` |
| `n3_ranked_terminal_local_u2000_seed2` | 3 | ranked | terminal | local, 4 inputs | 192000 | 2 | 900 | 89.5 | `0b6a708e61d4` |
| `n3_ranked_terminal_local_u2000_seed3` | 3 | ranked | terminal | local, 4 inputs | 192000 | 3 | 900 | 90.6 | `5c57577f49bb` |
| `n3_ranked_terminal_local_u2000_seed4` | 3 | ranked | terminal | local, 4 inputs | 192000 | 4 | 900 | 86.8 | `e56e47665128` |
| `n5_ranked_terminal_centralised_u2000_seed0` | 5 | ranked | terminal | centralised, 15 inputs | 192000 | 0 | 1604 | 162.8 | `5ca09d12ddd6` |
| `n5_ranked_terminal_centralised_u2000_seed1` | 5 | ranked | terminal | centralised, 15 inputs | 192000 | 1 | 1604 | 155.5 | `48acfed87282` |
| `n5_ranked_terminal_centralised_u2000_seed2` | 5 | ranked | terminal | centralised, 15 inputs | 192000 | 2 | 1604 | 149.6 | `3f5fa820420b` |
| `n5_ranked_terminal_centralised_u2000_seed3` | 5 | ranked | terminal | centralised, 15 inputs | 192000 | 3 | 1604 | 147.3 | `a12e660ee5fe` |
| `n5_ranked_terminal_centralised_u2000_seed4` | 5 | ranked | terminal | centralised, 15 inputs | 192000 | 4 | 1604 | 150.9 | `22c67c171cfe` |
| `n5_ranked_terminal_local_u2000_seed0` | 5 | ranked | terminal | local, 4 inputs | 192000 | 0 | 900 | 145.8 | `f390366cd7ba` |
| `n5_ranked_terminal_local_u2000_seed1` | 5 | ranked | terminal | local, 4 inputs | 192000 | 1 | 900 | 147.3 | `ca1ba5b84d83` |
| `n5_ranked_terminal_local_u2000_seed2` | 5 | ranked | terminal | local, 4 inputs | 192000 | 2 | 900 | 145.3 | `bbfe7bf6be13` |
| `n5_ranked_terminal_local_u2000_seed3` | 5 | ranked | terminal | local, 4 inputs | 192000 | 3 | 900 | 148.2 | `c32ac9cd86c7` |
| `n5_ranked_terminal_local_u2000_seed4` | 5 | ranked | terminal | local, 4 inputs | 192000 | 4 | 900 | 146.5 | `f4ccebdf55f1` |
| `n7_ranked_terminal_centralised_u2000_seed0` | 7 | ranked | terminal | centralised, 21 inputs | 192000 | 0 | 1988 | 229.7 | `8a06afdc50fc` |
| `n7_ranked_terminal_centralised_u2000_seed1` | 7 | ranked | terminal | centralised, 21 inputs | 192000 | 1 | 1988 | 226.3 | `bf2cffe70a69` |
| `n7_ranked_terminal_centralised_u2000_seed2` | 7 | ranked | terminal | centralised, 21 inputs | 192000 | 2 | 1988 | 222.7 | `b1377c587144` |
| `n7_ranked_terminal_centralised_u2000_seed3` | 7 | ranked | terminal | centralised, 21 inputs | 192000 | 3 | 1988 | 227.0 | `2341f2d18c17` |
| `n7_ranked_terminal_centralised_u2000_seed4` | 7 | ranked | terminal | centralised, 21 inputs | 192000 | 4 | 1988 | 227.4 | `00dbb8404da6` |
| `n7_ranked_terminal_local_u2000_seed0` | 7 | ranked | terminal | local, 4 inputs | 192000 | 0 | 900 | 219.9 | `172c247a0443` |
| `n7_ranked_terminal_local_u2000_seed1` | 7 | ranked | terminal | local, 4 inputs | 192000 | 1 | 900 | 217.6 | `a156f7c1a3b9` |
| `n7_ranked_terminal_local_u2000_seed2` | 7 | ranked | terminal | local, 4 inputs | 192000 | 2 | 900 | 219.1 | `c6563ebd2fb1` |
| `n7_ranked_terminal_local_u2000_seed3` | 7 | ranked | terminal | local, 4 inputs | 192000 | 3 | 900 | 209.9 | `45ecb98e6387` |
| `n7_ranked_terminal_local_u2000_seed4` | 7 | ranked | terminal | local, 4 inputs | 192000 | 4 | 900 | 181.9 | `45e9406c6878` |

### `03-agent-architecture/step2-ppo`

| Run | N | Seating | Reward | Critic input | Training episodes | Seed | Parameters | Seconds | weights_sha256 |
|---|---|---|---|---|---|---|---|---|---|
| `n2_ranked_terminal_ppo_no_entropy_u40_seed0` | 2 | ranked | terminal | local, 4 inputs | 2560 | 2000 | 900 | 1.2 | `eb10af653837` |
| `n2_ranked_terminal_ppo_u40_seed0` | 2 | ranked | terminal | local, 4 inputs | 2560 | 2000 | 900 | 2.3 | `8c8096a9cd66` |
| `sweep_n2_ranked_terminal_ppo_u10_seed0` | 2 | ranked | terminal | local, 4 inputs | 640 | 2000 | 900 | 0.2 | `623eb92cb600` |
| `sweep_n2_ranked_terminal_ppo_u10_seed1` | 2 | ranked | terminal | local, 4 inputs | 640 | 2001 | 900 | 0.2 | `874923efe7f7` |
| `sweep_n2_ranked_terminal_ppo_u10_seed2` | 2 | ranked | terminal | local, 4 inputs | 640 | 2002 | 900 | 0.2 | `f089a4fe2483` |
| `sweep_n2_ranked_terminal_ppo_u10_seed3` | 2 | ranked | terminal | local, 4 inputs | 640 | 2003 | 900 | 0.2 | `dd0bb7ac584d` |
| `sweep_n2_ranked_terminal_ppo_u20_seed0` | 2 | ranked | terminal | local, 4 inputs | 1280 | 2000 | 900 | 0.4 | `3dca08a0fc22` |
| `sweep_n2_ranked_terminal_ppo_u20_seed1` | 2 | ranked | terminal | local, 4 inputs | 1280 | 2001 | 900 | 0.5 | `7ef14b3eb6f1` |
| `sweep_n2_ranked_terminal_ppo_u20_seed2` | 2 | ranked | terminal | local, 4 inputs | 1280 | 2002 | 900 | 0.5 | `30bf70324347` |
| `sweep_n2_ranked_terminal_ppo_u20_seed3` | 2 | ranked | terminal | local, 4 inputs | 1280 | 2003 | 900 | 0.5 | `84ef45d4f362` |
| `sweep_n2_ranked_terminal_ppo_u40_seed0` | 2 | ranked | terminal | local, 4 inputs | 2560 | 2000 | 900 | 1.0 | `8c8096a9cd66` |
| `sweep_n2_ranked_terminal_ppo_u40_seed1` | 2 | ranked | terminal | local, 4 inputs | 2560 | 2001 | 900 | 1.2 | `e8dde015a038` |
| `sweep_n2_ranked_terminal_ppo_u40_seed2` | 2 | ranked | terminal | local, 4 inputs | 2560 | 2002 | 900 | 1.1 | `573b02985387` |
| `sweep_n2_ranked_terminal_ppo_u40_seed3` | 2 | ranked | terminal | local, 4 inputs | 2560 | 2003 | 900 | 1.2 | `a26f5ca7f571` |
| `sweep_n2_ranked_terminal_ppo_u5_seed0` | 2 | ranked | terminal | local, 4 inputs | 320 | 2000 | 900 | 0.1 | `7d744f6fca35` |
| `sweep_n2_ranked_terminal_ppo_u5_seed1` | 2 | ranked | terminal | local, 4 inputs | 320 | 2001 | 900 | 0.2 | `a3d226b16a54` |
| `sweep_n2_ranked_terminal_ppo_u5_seed2` | 2 | ranked | terminal | local, 4 inputs | 320 | 2002 | 900 | 0.1 | `57b781364df0` |
| `sweep_n2_ranked_terminal_ppo_u5_seed3` | 2 | ranked | terminal | local, 4 inputs | 320 | 2003 | 900 | 0.1 | `5e596f121210` |
| `sweep_n2_ranked_terminal_reinforce_e1280_seed0` | 2 | ranked | terminal | none | 1280 | 2000 | 515 | 0.4 | `7d270223c29e` |
| `sweep_n2_ranked_terminal_reinforce_e1280_seed1` | 2 | ranked | terminal | none | 1280 | 2001 | 515 | 0.5 | `43a6a32887bc` |
| `sweep_n2_ranked_terminal_reinforce_e1280_seed2` | 2 | ranked | terminal | none | 1280 | 2002 | 515 | 0.5 | `ffc1e7393e4b` |
| `sweep_n2_ranked_terminal_reinforce_e1280_seed3` | 2 | ranked | terminal | none | 1280 | 2003 | 515 | 0.5 | `2d77c8b5b92c` |
| `sweep_n2_ranked_terminal_reinforce_e2560_seed0` | 2 | ranked | terminal | none | 2560 | 2000 | 515 | 0.9 | `ebf893d5fccb` |
| `sweep_n2_ranked_terminal_reinforce_e2560_seed1` | 2 | ranked | terminal | none | 2560 | 2001 | 515 | 1.1 | `abe890e21b36` |
| `sweep_n2_ranked_terminal_reinforce_e2560_seed2` | 2 | ranked | terminal | none | 2560 | 2002 | 515 | 1.1 | `31b6370f0cee` |
| `sweep_n2_ranked_terminal_reinforce_e2560_seed3` | 2 | ranked | terminal | none | 2560 | 2003 | 515 | 1.2 | `f9b8d752dcff` |
| `sweep_n2_ranked_terminal_reinforce_e320_seed0` | 2 | ranked | terminal | none | 320 | 2000 | 515 | 0.1 | `58e62cdb7696` |
| `sweep_n2_ranked_terminal_reinforce_e320_seed1` | 2 | ranked | terminal | none | 320 | 2001 | 515 | 0.1 | `930719128a75` |
| `sweep_n2_ranked_terminal_reinforce_e320_seed2` | 2 | ranked | terminal | none | 320 | 2002 | 515 | 0.1 | `e77ea7da8935` |
| `sweep_n2_ranked_terminal_reinforce_e320_seed3` | 2 | ranked | terminal | none | 320 | 2003 | 515 | 0.1 | `224ebae291aa` |
| `sweep_n2_ranked_terminal_reinforce_e640_seed0` | 2 | ranked | terminal | none | 640 | 2000 | 515 | 0.2 | `ce7ae6b5f322` |
| `sweep_n2_ranked_terminal_reinforce_e640_seed1` | 2 | ranked | terminal | none | 640 | 2001 | 515 | 0.2 | `7ce1eaddf92a` |
| `sweep_n2_ranked_terminal_reinforce_e640_seed2` | 2 | ranked | terminal | none | 640 | 2002 | 515 | 0.3 | `0ef254db9173` |
| `sweep_n2_ranked_terminal_reinforce_e640_seed3` | 2 | ranked | terminal | none | 640 | 2003 | 515 | 0.2 | `b99c46886db3` |

### `03-agent-architecture/step3-mappo`

| Run | N | Seating | Reward | Critic input | Training episodes | Seed | Parameters | Seconds | weights_sha256 |
|---|---|---|---|---|---|---|---|---|---|
| `n3_ranked_terminal_centralised_u400_seed0` | 3 | ranked | terminal | centralised, 9 inputs | 38400 | 0 | 1220 | 25.5 | `f112e0745585` |

### `03-agent-architecture/step5-targeting`

| Run | N | Seating | Reward | Critic input | Training episodes | Seed | Parameters | Seconds | weights_sha256 |
|---|---|---|---|---|---|---|---|---|---|
| `n3_ranked_terminal_centralised_u400_seed0` | 3 | ranked | terminal | centralised, 9 inputs | 38400 | 0 | 1220 | 24.7 | `f112e0745585` |

### `03-agent-architecture/step6-n5-gate`

| Run | N | Seating | Reward | Critic input | Training episodes | Seed | Parameters | Seconds | weights_sha256 |
|---|---|---|---|---|---|---|---|---|---|
| `n5_ranked_terminal_centralised_u2000_seed0` | 5 | ranked | terminal | centralised, 15 inputs | 192000 | 0 | 1604 | 135.5 | `5ca09d12ddd6` |
| `n5_ranked_terminal_centralised_u2000_seed1` | 5 | ranked | terminal | centralised, 15 inputs | 192000 | 1 | 1604 | 167.1 | `48acfed87282` |
| `n5_ranked_terminal_centralised_u2000_seed2` | 5 | ranked | terminal | centralised, 15 inputs | 192000 | 2 | 1604 | 170.5 | `3f5fa820420b` |
| `n5_ranked_terminal_centralised_u2000_seed3` | 5 | ranked | terminal | centralised, 15 inputs | 192000 | 3 | 1604 | 174.3 | `a12e660ee5fe` |
| `n5_ranked_terminal_centralised_u2000_seed4` | 5 | ranked | terminal | centralised, 15 inputs | 192000 | 4 | 1604 | 178.3 | `22c67c171cfe` |

### `03-agent-architecture/step7-seating`

| Run | N | Seating | Reward | Critic input | Training episodes | Seed | Parameters | Seconds | weights_sha256 |
|---|---|---|---|---|---|---|---|---|---|
| `n5_random_terminal_centralised_u2000_seed0` | 5 | random | terminal | centralised, 15 inputs | 192000 | 0 | 1604 | 181.2 | `5aaefbc69b83` |
| `n5_random_terminal_centralised_u2000_seed1` | 5 | random | terminal | centralised, 15 inputs | 192000 | 1 | 1604 | 186.1 | `b31f6d2dbfb4` |
| `n5_random_terminal_centralised_u2000_seed2` | 5 | random | terminal | centralised, 15 inputs | 192000 | 2 | 1604 | 171.9 | `c958f824b379` |
| `n5_ranked_terminal_centralised_u2000_seed0` | 5 | ranked | terminal | centralised, 15 inputs | 192000 | 0 | 1604 | 136.1 | `5ca09d12ddd6` |
| `n5_ranked_terminal_centralised_u2000_seed1` | 5 | ranked | terminal | centralised, 15 inputs | 192000 | 1 | 1604 | 167.5 | `48acfed87282` |
| `n5_ranked_terminal_centralised_u2000_seed2` | 5 | ranked | terminal | centralised, 15 inputs | 192000 | 2 | 1604 | 171.2 | `3f5fa820420b` |

### `03-agent-architecture/step8-reward-profile`

| Run | N | Seating | Reward | Critic input | Training episodes | Seed | Parameters | Seconds | weights_sha256 |
|---|---|---|---|---|---|---|---|---|---|
| `n5_ranked_shaped_centralised_u2000_seed0` | 5 | ranked | shaped | centralised, 15 inputs | 192000 | 0 | 1604 | 273.1 | `f9ab2f97f193` |
| `n5_ranked_shaped_centralised_u2000_seed1` | 5 | ranked | shaped | centralised, 15 inputs | 192000 | 1 | 1604 | 273.0 | `205bf17466ee` |
| `n5_ranked_shaped_centralised_u2000_seed2` | 5 | ranked | shaped | centralised, 15 inputs | 192000 | 2 | 1604 | 260.8 | `15732dedb81b` |
| `n5_ranked_terminal_centralised_u2000_seed0` | 5 | ranked | terminal | centralised, 15 inputs | 192000 | 0 | 1604 | 276.7 | `5ca09d12ddd6` |
| `n5_ranked_terminal_centralised_u2000_seed1` | 5 | ranked | terminal | centralised, 15 inputs | 192000 | 1 | 1604 | 283.8 | `48acfed87282` |
| `n5_ranked_terminal_centralised_u2000_seed2` | 5 | ranked | terminal | centralised, 15 inputs | 192000 | 2 | 1604 | 283.3 | `3f5fa820420b` |

### `03-agent-architecture/step9-player-count`

| Run | N | Seating | Reward | Critic input | Training episodes | Seed | Parameters | Seconds | weights_sha256 |
|---|---|---|---|---|---|---|---|---|---|
| `n3_ranked_terminal_centralised_u2000_seed0` | 3 | ranked | terminal | centralised, 9 inputs | 192000 | 0 | 1220 | 133.5 | `f29ae0f9b5ff` |
| `n3_ranked_terminal_centralised_u2000_seed1` | 3 | ranked | terminal | centralised, 9 inputs | 192000 | 1 | 1220 | 151.8 | `03fe4e8e9674` |
| `n3_ranked_terminal_centralised_u2000_seed2` | 3 | ranked | terminal | centralised, 9 inputs | 192000 | 2 | 1220 | 147.0 | `470df0cf9139` |
| `n5_ranked_terminal_centralised_u2000_seed0` | 5 | ranked | terminal | centralised, 15 inputs | 192000 | 0 | 1604 | 285.9 | `5ca09d12ddd6` |
| `n5_ranked_terminal_centralised_u2000_seed1` | 5 | ranked | terminal | centralised, 15 inputs | 192000 | 1 | 1604 | 273.6 | `48acfed87282` |
| `n5_ranked_terminal_centralised_u2000_seed2` | 5 | ranked | terminal | centralised, 15 inputs | 192000 | 2 | 1604 | 270.1 | `3f5fa820420b` |
| `n7_ranked_terminal_centralised_u2000_seed0` | 7 | ranked | terminal | centralised, 21 inputs | 192000 | 0 | 1988 | 379.3 | `8a06afdc50fc` |
| `n7_ranked_terminal_centralised_u2000_seed1` | 7 | ranked | terminal | centralised, 21 inputs | 192000 | 1 | 1988 | 326.7 | `bf2cffe70a69` |
| `n7_ranked_terminal_centralised_u2000_seed2` | 7 | ranked | terminal | centralised, 21 inputs | 192000 | 2 | 1988 | 319.8 | `b1377c587144` |
| `n7_ranked_terminal_centralised_u4000_seed0` | 7 | ranked | terminal | centralised, 21 inputs | 384000 | 0 | 1988 | 623.4 | `ac6bc32bd54a` |

### `04-scaling/step1-train-any-n`

| Run | N | Seating | Reward | Critic input | Training episodes | Seed | Parameters | Seconds | weights_sha256 |
|---|---|---|---|---|---|---|---|---|---|
| `n10_ranked_terminal_centralised_u2000_seed0` | 10 | ranked | terminal | centralised, 30 inputs | 192000 | 0 | 2564 | 338.5 | `1e83c5b58816` |
| `n10_ranked_terminal_centralised_u2000_seed1` | 10 | ranked | terminal | centralised, 30 inputs | 192000 | 1 | 2564 | 342.4 | `73d6ea09a453` |
| `n10_ranked_terminal_centralised_u2000_seed2` | 10 | ranked | terminal | centralised, 30 inputs | 192000 | 2 | 2564 | 347.9 | `b4fdaf6fa76d` |
| `n15_ranked_terminal_centralised_u2000_seed0` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 0 | 3524 | 550.0 | `e064a9eb5460` |
| `n15_ranked_terminal_centralised_u2000_seed1` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 1 | 3524 | 559.6 | `dd04e571ea5f` |
| `n15_ranked_terminal_centralised_u2000_seed2` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 2 | 3524 | 551.1 | `faf1813f09e9` |
| `n3_ranked_terminal_centralised_u2000_seed0` | 3 | ranked | terminal | centralised, 9 inputs | 192000 | 0 | 1220 | 64.4 | `f29ae0f9b5ff` |
| `n3_ranked_terminal_centralised_u2000_seed1` | 3 | ranked | terminal | centralised, 9 inputs | 192000 | 1 | 1220 | 75.6 | `03fe4e8e9674` |
| `n3_ranked_terminal_centralised_u2000_seed2` | 3 | ranked | terminal | centralised, 9 inputs | 192000 | 2 | 1220 | 87.0 | `470df0cf9139` |
| `n5_ranked_terminal_centralised_u2000_seed0` | 5 | ranked | terminal | centralised, 15 inputs | 192000 | 0 | 1604 | 163.1 | `5ca09d12ddd6` |
| `n5_ranked_terminal_centralised_u2000_seed1` | 5 | ranked | terminal | centralised, 15 inputs | 192000 | 1 | 1604 | 170.2 | `48acfed87282` |
| `n5_ranked_terminal_centralised_u2000_seed2` | 5 | ranked | terminal | centralised, 15 inputs | 192000 | 2 | 1604 | 177.0 | `3f5fa820420b` |
| `n7_ranked_terminal_centralised_u2000_seed0` | 7 | ranked | terminal | centralised, 21 inputs | 192000 | 0 | 1988 | 245.0 | `8a06afdc50fc` |
| `n7_ranked_terminal_centralised_u2000_seed1` | 7 | ranked | terminal | centralised, 21 inputs | 192000 | 1 | 1988 | 226.7 | `bf2cffe70a69` |
| `n7_ranked_terminal_centralised_u2000_seed2` | 7 | ranked | terminal | centralised, 21 inputs | 192000 | 2 | 1988 | 219.0 | `b1377c587144` |

### `04-scaling/step2-switch-test`

| Run | N | Seating | Reward | Critic input | Training episodes | Seed | Parameters | Seconds | weights_sha256 |
|---|---|---|---|---|---|---|---|---|---|
| `n10_ranked_terminal_centralised_u2000_seed0` | 10 | ranked | terminal | centralised, 30 inputs | 192000 | 0 | 2564 | 583.5 | `1e83c5b58816` |
| `n10_ranked_terminal_centralised_u2000_seed1` | 10 | ranked | terminal | centralised, 30 inputs | 192000 | 1 | 2564 | 505.8 | `73d6ea09a453` |
| `n10_ranked_terminal_centralised_u2000_seed2` | 10 | ranked | terminal | centralised, 30 inputs | 192000 | 2 | 2564 | 469.7 | `b4fdaf6fa76d` |
| `n15_ranked_terminal_centralised_u2000_seed0` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 0 | 3524 | 738.4 | `e064a9eb5460` |
| `n15_ranked_terminal_centralised_u2000_seed1` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 1 | 3524 | 724.3 | `dd04e571ea5f` |
| `n15_ranked_terminal_centralised_u2000_seed2` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 2 | 3524 | 732.1 | `faf1813f09e9` |
| `n5_ranked_terminal_centralised_u2000_seed0` | 5 | ranked | terminal | centralised, 15 inputs | 192000 | 0 | 1604 | 258.9 | `5ca09d12ddd6` |
| `n5_ranked_terminal_centralised_u2000_seed1` | 5 | ranked | terminal | centralised, 15 inputs | 192000 | 1 | 1604 | 269.3 | `48acfed87282` |
| `n5_ranked_terminal_centralised_u2000_seed2` | 5 | ranked | terminal | centralised, 15 inputs | 192000 | 2 | 1604 | 271.3 | `3f5fa820420b` |

### `04-scaling/step3-targeting`

| Run | N | Seating | Reward | Critic input | Training episodes | Seed | Parameters | Seconds | weights_sha256 |
|---|---|---|---|---|---|---|---|---|---|
| `n10_ranked_terminal_centralised_u2000_seed0` | 10 | ranked | terminal | centralised, 30 inputs | 192000 | 0 | 2564 | 583.7 | `1e83c5b58816` |
| `n10_ranked_terminal_centralised_u2000_seed1` | 10 | ranked | terminal | centralised, 30 inputs | 192000 | 1 | 2564 | 505.4 | `73d6ea09a453` |
| `n10_ranked_terminal_centralised_u2000_seed2` | 10 | ranked | terminal | centralised, 30 inputs | 192000 | 2 | 2564 | 472.7 | `b4fdaf6fa76d` |
| `n15_ranked_terminal_centralised_u2000_seed0` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 0 | 3524 | 738.2 | `e064a9eb5460` |
| `n15_ranked_terminal_centralised_u2000_seed1` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 1 | 3524 | 730.9 | `dd04e571ea5f` |
| `n15_ranked_terminal_centralised_u2000_seed2` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 2 | 3524 | 733.6 | `faf1813f09e9` |
| `n5_ranked_terminal_centralised_u2000_seed0` | 5 | ranked | terminal | centralised, 15 inputs | 192000 | 0 | 1604 | 259.0 | `5ca09d12ddd6` |
| `n5_ranked_terminal_centralised_u2000_seed1` | 5 | ranked | terminal | centralised, 15 inputs | 192000 | 1 | 1604 | 269.7 | `48acfed87282` |
| `n5_ranked_terminal_centralised_u2000_seed2` | 5 | ranked | terminal | centralised, 15 inputs | 192000 | 2 | 1604 | 270.9 | `3f5fa820420b` |

### `04-scaling/step4-continuity`

| Run | N | Seating | Reward | Critic input | Training episodes | Seed | Parameters | Seconds | weights_sha256 |
|---|---|---|---|---|---|---|---|---|---|
| `n10_ranked_terminal_centralised_u2000_seed0` | 10 | ranked | terminal | centralised, 30 inputs | 192000 | 0 | 2564 | 341.7 | `1e83c5b58816` |
| `n10_ranked_terminal_centralised_u2000_seed1` | 10 | ranked | terminal | centralised, 30 inputs | 192000 | 1 | 2564 | 345.2 | `73d6ea09a453` |
| `n10_ranked_terminal_centralised_u2000_seed2` | 10 | ranked | terminal | centralised, 30 inputs | 192000 | 2 | 2564 | 350.4 | `b4fdaf6fa76d` |
| `n15_ranked_terminal_centralised_u2000_seed0` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 0 | 3524 | 552.7 | `e064a9eb5460` |
| `n15_ranked_terminal_centralised_u2000_seed1` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 1 | 3524 | 562.7 | `dd04e571ea5f` |
| `n15_ranked_terminal_centralised_u2000_seed2` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 2 | 3524 | 554.7 | `faf1813f09e9` |
| `n3_ranked_terminal_centralised_u2000_seed0` | 3 | ranked | terminal | centralised, 9 inputs | 192000 | 0 | 1220 | 64.9 | `f29ae0f9b5ff` |
| `n3_ranked_terminal_centralised_u2000_seed1` | 3 | ranked | terminal | centralised, 9 inputs | 192000 | 1 | 1220 | 76.2 | `03fe4e8e9674` |
| `n3_ranked_terminal_centralised_u2000_seed2` | 3 | ranked | terminal | centralised, 9 inputs | 192000 | 2 | 1220 | 87.8 | `470df0cf9139` |
| `n5_ranked_terminal_centralised_u2000_seed0` | 5 | ranked | terminal | centralised, 15 inputs | 192000 | 0 | 1604 | 164.0 | `5ca09d12ddd6` |
| `n5_ranked_terminal_centralised_u2000_seed1` | 5 | ranked | terminal | centralised, 15 inputs | 192000 | 1 | 1604 | 171.1 | `48acfed87282` |
| `n5_ranked_terminal_centralised_u2000_seed2` | 5 | ranked | terminal | centralised, 15 inputs | 192000 | 2 | 1604 | 178.0 | `3f5fa820420b` |
| `n7_ranked_terminal_centralised_u2000_seed0` | 7 | ranked | terminal | centralised, 21 inputs | 192000 | 0 | 1988 | 246.5 | `8a06afdc50fc` |
| `n7_ranked_terminal_centralised_u2000_seed1` | 7 | ranked | terminal | centralised, 21 inputs | 192000 | 1 | 1988 | 227.9 | `bf2cffe70a69` |
| `n7_ranked_terminal_centralised_u2000_seed2` | 7 | ranked | terminal | centralised, 21 inputs | 192000 | 2 | 1988 | 220.8 | `b1377c587144` |

### `04-scaling/step5-reward-profile`

| Run | N | Seating | Reward | Critic input | Training episodes | Seed | Parameters | Seconds | weights_sha256 |
|---|---|---|---|---|---|---|---|---|---|
| `n10_ranked_shaped_centralised_u2000_seed0` | 10 | ranked | shaped | centralised, 30 inputs | 192000 | 0 | 2564 | 624.1 | `cb8ca3823f40` |
| `n10_ranked_shaped_centralised_u2000_seed1` | 10 | ranked | shaped | centralised, 30 inputs | 192000 | 1 | 2564 | 643.5 | `64bf16e2307a` |
| `n10_ranked_shaped_centralised_u2000_seed2` | 10 | ranked | shaped | centralised, 30 inputs | 192000 | 2 | 2564 | 622.6 | `205915d0defd` |
| `n10_ranked_terminal_centralised_u2000_seed0` | 10 | ranked | terminal | centralised, 30 inputs | 192000 | 0 | 2564 | 457.1 | `1e83c5b58816` |
| `n10_ranked_terminal_centralised_u2000_seed1` | 10 | ranked | terminal | centralised, 30 inputs | 192000 | 1 | 2564 | 629.5 | `73d6ea09a453` |
| `n10_ranked_terminal_centralised_u2000_seed2` | 10 | ranked | terminal | centralised, 30 inputs | 192000 | 2 | 2564 | 637.9 | `b4fdaf6fa76d` |
| `n15_ranked_shaped_centralised_u2000_seed0` | 15 | ranked | shaped | centralised, 45 inputs | 192000 | 0 | 3524 | 446.8 | `613411cfbfff` |
| `n15_ranked_shaped_centralised_u2000_seed1` | 15 | ranked | shaped | centralised, 45 inputs | 192000 | 1 | 3524 | 403.3 | `4d81a7da37ac` |
| `n15_ranked_shaped_centralised_u2000_seed2` | 15 | ranked | shaped | centralised, 45 inputs | 192000 | 2 | 3524 | 473.0 | `2b3dd379abf0` |
| `n15_ranked_terminal_centralised_u2000_seed0` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 0 | 3524 | 956.7 | `e064a9eb5460` |
| `n15_ranked_terminal_centralised_u2000_seed1` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 1 | 3524 | 1018.2 | `dd04e571ea5f` |
| `n15_ranked_terminal_centralised_u2000_seed2` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 2 | 3524 | 995.9 | `faf1813f09e9` |

### `04-scaling/step6-seating`

| Run | N | Seating | Reward | Critic input | Training episodes | Seed | Parameters | Seconds | weights_sha256 |
|---|---|---|---|---|---|---|---|---|---|
| `n10_random_terminal_centralised_u2000_seed0` | 10 | random | terminal | centralised, 30 inputs | 192000 | 0 | 2564 | 357.8 | `7edda291f12f` |
| `n10_random_terminal_centralised_u2000_seed1` | 10 | random | terminal | centralised, 30 inputs | 192000 | 1 | 2564 | 352.9 | `971b28ae1cc1` |
| `n10_random_terminal_centralised_u2000_seed2` | 10 | random | terminal | centralised, 30 inputs | 192000 | 2 | 2564 | 366.9 | `e32ff30b7264` |
| `n10_ranked_terminal_centralised_u2000_seed0` | 10 | ranked | terminal | centralised, 30 inputs | 192000 | 0 | 2564 | 340.9 | `1e83c5b58816` |
| `n10_ranked_terminal_centralised_u2000_seed1` | 10 | ranked | terminal | centralised, 30 inputs | 192000 | 1 | 2564 | 394.5 | `73d6ea09a453` |
| `n10_ranked_terminal_centralised_u2000_seed2` | 10 | ranked | terminal | centralised, 30 inputs | 192000 | 2 | 2564 | 372.0 | `b4fdaf6fa76d` |
| `n15_random_terminal_centralised_u2000_seed0` | 15 | random | terminal | centralised, 45 inputs | 192000 | 0 | 3524 | 533.6 | `e45a9a69583c` |
| `n15_random_terminal_centralised_u2000_seed1` | 15 | random | terminal | centralised, 45 inputs | 192000 | 1 | 3524 | 443.6 | `42eacd1189cb` |
| `n15_random_terminal_centralised_u2000_seed2` | 15 | random | terminal | centralised, 45 inputs | 192000 | 2 | 3524 | 439.3 | `e65d696745d1` |
| `n15_ranked_terminal_centralised_u2000_seed0` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 0 | 3524 | 559.5 | `e064a9eb5460` |
| `n15_ranked_terminal_centralised_u2000_seed1` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 1 | 3524 | 561.6 | `dd04e571ea5f` |
| `n15_ranked_terminal_centralised_u2000_seed2` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 2 | 3524 | 559.9 | `faf1813f09e9` |

### `04-scaling/step7-holding-fire`

| Run | N | Seating | Reward | Critic input | Training episodes | Seed | Parameters | Seconds | weights_sha256 |
|---|---|---|---|---|---|---|---|---|---|
| `n15_ranked_terminal_centralised_u2000_seed0` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 0 | 3524 | 568.5 | `e064a9eb5460` |
| `n15_ranked_terminal_centralised_u2000_seed1` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 1 | 3524 | 600.2 | `dd04e571ea5f` |
| `n15_ranked_terminal_centralised_u2000_seed2` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 2 | 3524 | 540.6 | `faf1813f09e9` |
| `n15_ranked_terminal_centralised_u2000_seed3` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 3 | 3524 | 551.7 | `0d5cd3dafa11` |
| `n15_ranked_terminal_centralised_u2000_seed4` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 4 | 3524 | 582.9 | `bd2e1c73ac04` |
| `n15_ranked_terminal_centralised_u2000_seed5` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 5 | 3524 | 551.8 | `82583b3fb0f4` |
| `n15_ranked_terminal_centralised_u2000_seed6` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 6 | 3524 | 559.1 | `e1fa19b581f1` |
| `n15_ranked_terminal_centralised_u2000_seed7` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 7 | 3524 | 493.7 | `37af87e87a02` |
| `n15_ranked_terminal_centralised_u2000_seed8` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 8 | 3524 | 437.7 | `18f6a9ee59f9` |
| `n15_ranked_terminal_centralised_u2000_seed9` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 9 | 3524 | 414.1 | `40c18ec684cf` |

### `04-scaling/step8-critic-ablation`

| Run | N | Seating | Reward | Critic input | Training episodes | Seed | Parameters | Seconds | weights_sha256 |
|---|---|---|---|---|---|---|---|---|---|
| `n10_ranked_terminal_centralised_u2000_seed0` | 10 | ranked | terminal | centralised, 30 inputs | 192000 | 0 | 2564 | 584.3 | `1e83c5b58816` |
| `n10_ranked_terminal_centralised_u2000_seed1` | 10 | ranked | terminal | centralised, 30 inputs | 192000 | 1 | 2564 | 600.0 | `73d6ea09a453` |
| `n10_ranked_terminal_centralised_u2000_seed2` | 10 | ranked | terminal | centralised, 30 inputs | 192000 | 2 | 2564 | 597.6 | `b4fdaf6fa76d` |
| `n10_ranked_terminal_centralised_u2000_seed3` | 10 | ranked | terminal | centralised, 30 inputs | 192000 | 3 | 2564 | 633.2 | `ea47cc0bd630` |
| `n10_ranked_terminal_centralised_u2000_seed4` | 10 | ranked | terminal | centralised, 30 inputs | 192000 | 4 | 2564 | 647.9 | `b56f7c2ccf86` |
| `n10_ranked_terminal_local_u2000_seed0` | 10 | ranked | terminal | local, 4 inputs | 192000 | 0 | 900 | 593.9 | `b1bccc57b322` |
| `n10_ranked_terminal_local_u2000_seed1` | 10 | ranked | terminal | local, 4 inputs | 192000 | 1 | 900 | 588.0 | `0eec14c9c390` |
| `n10_ranked_terminal_local_u2000_seed2` | 10 | ranked | terminal | local, 4 inputs | 192000 | 2 | 900 | 606.9 | `e18cc1f3e9a0` |
| `n10_ranked_terminal_local_u2000_seed3` | 10 | ranked | terminal | local, 4 inputs | 192000 | 3 | 900 | 615.2 | `2099e87b8b01` |
| `n10_ranked_terminal_local_u2000_seed4` | 10 | ranked | terminal | local, 4 inputs | 192000 | 4 | 900 | 638.5 | `93f14c233ab1` |
| `n15_ranked_terminal_centralised_u2000_seed0` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 0 | 3524 | 885.6 | `e064a9eb5460` |
| `n15_ranked_terminal_centralised_u2000_seed1` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 1 | 3524 | 876.8 | `dd04e571ea5f` |
| `n15_ranked_terminal_centralised_u2000_seed2` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 2 | 3524 | 869.5 | `faf1813f09e9` |
| `n15_ranked_terminal_centralised_u2000_seed3` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 3 | 3524 | 868.9 | `0d5cd3dafa11` |
| `n15_ranked_terminal_centralised_u2000_seed4` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 4 | 3524 | 931.4 | `bd2e1c73ac04` |
| `n15_ranked_terminal_local_u2000_seed0` | 15 | ranked | terminal | local, 4 inputs | 192000 | 0 | 900 | 823.1 | `649594d5bb61` |
| `n15_ranked_terminal_local_u2000_seed1` | 15 | ranked | terminal | local, 4 inputs | 192000 | 1 | 900 | 840.1 | `cc8cf9955407` |
| `n15_ranked_terminal_local_u2000_seed2` | 15 | ranked | terminal | local, 4 inputs | 192000 | 2 | 900 | 815.7 | `673f270dbd9f` |
| `n15_ranked_terminal_local_u2000_seed3` | 15 | ranked | terminal | local, 4 inputs | 192000 | 3 | 900 | 830.6 | `efc41c8872a3` |
| `n15_ranked_terminal_local_u2000_seed4` | 15 | ranked | terminal | local, 4 inputs | 192000 | 4 | 900 | 802.7 | `fc5db23dd656` |

### `04-scaling/step9-direction`

| Run | N | Seating | Reward | Critic input | Training episodes | Seed | Parameters | Seconds | weights_sha256 |
|---|---|---|---|---|---|---|---|---|---|
| `n10_ranked_terminal_centralised_u2000_seed0` | 10 | ranked | terminal | centralised, 30 inputs | 192000 | 0 | 2564 | 591.9 | `1e83c5b58816` |
| `n10_ranked_terminal_centralised_u2000_seed1` | 10 | ranked | terminal | centralised, 30 inputs | 192000 | 1 | 2564 | 596.0 | `73d6ea09a453` |
| `n10_ranked_terminal_centralised_u2000_seed10` | 10 | ranked | terminal | centralised, 30 inputs | 192000 | 10 | 2564 | 586.0 | `c765e23f8304` |
| `n10_ranked_terminal_centralised_u2000_seed11` | 10 | ranked | terminal | centralised, 30 inputs | 192000 | 11 | 2564 | 581.6 | `329e155a792c` |
| `n10_ranked_terminal_centralised_u2000_seed12` | 10 | ranked | terminal | centralised, 30 inputs | 192000 | 12 | 2564 | 599.7 | `da88141988ae` |
| `n10_ranked_terminal_centralised_u2000_seed13` | 10 | ranked | terminal | centralised, 30 inputs | 192000 | 13 | 2564 | 590.3 | `ea519a748031` |
| `n10_ranked_terminal_centralised_u2000_seed14` | 10 | ranked | terminal | centralised, 30 inputs | 192000 | 14 | 2564 | 607.9 | `a1c55ab80935` |
| `n10_ranked_terminal_centralised_u2000_seed15` | 10 | ranked | terminal | centralised, 30 inputs | 192000 | 15 | 2564 | 610.5 | `93a7505a64f8` |
| `n10_ranked_terminal_centralised_u2000_seed16` | 10 | ranked | terminal | centralised, 30 inputs | 192000 | 16 | 2564 | 612.7 | `8d515b4d18ae` |
| `n10_ranked_terminal_centralised_u2000_seed17` | 10 | ranked | terminal | centralised, 30 inputs | 192000 | 17 | 2564 | 607.3 | `fe8d513ed6b5` |
| `n10_ranked_terminal_centralised_u2000_seed18` | 10 | ranked | terminal | centralised, 30 inputs | 192000 | 18 | 2564 | 609.0 | `d52b4457d454` |
| `n10_ranked_terminal_centralised_u2000_seed19` | 10 | ranked | terminal | centralised, 30 inputs | 192000 | 19 | 2564 | 646.4 | `8f0868220067` |
| `n10_ranked_terminal_centralised_u2000_seed2` | 10 | ranked | terminal | centralised, 30 inputs | 192000 | 2 | 2564 | 601.1 | `b4fdaf6fa76d` |
| `n10_ranked_terminal_centralised_u2000_seed3` | 10 | ranked | terminal | centralised, 30 inputs | 192000 | 3 | 2564 | 602.2 | `ea47cc0bd630` |
| `n10_ranked_terminal_centralised_u2000_seed4` | 10 | ranked | terminal | centralised, 30 inputs | 192000 | 4 | 2564 | 619.7 | `b56f7c2ccf86` |
| `n10_ranked_terminal_centralised_u2000_seed5` | 10 | ranked | terminal | centralised, 30 inputs | 192000 | 5 | 2564 | 607.6 | `dd82a6e2c3c4` |
| `n10_ranked_terminal_centralised_u2000_seed6` | 10 | ranked | terminal | centralised, 30 inputs | 192000 | 6 | 2564 | 610.3 | `a2223d5dd3d4` |
| `n10_ranked_terminal_centralised_u2000_seed7` | 10 | ranked | terminal | centralised, 30 inputs | 192000 | 7 | 2564 | 603.5 | `c9362ef562a2` |
| `n10_ranked_terminal_centralised_u2000_seed8` | 10 | ranked | terminal | centralised, 30 inputs | 192000 | 8 | 2564 | 592.8 | `7a4756c903a4` |
| `n10_ranked_terminal_centralised_u2000_seed9` | 10 | ranked | terminal | centralised, 30 inputs | 192000 | 9 | 2564 | 612.2 | `fe83bc0a99b4` |
| `n15_ranked_terminal_centralised_u2000_seed0` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 0 | 3524 | 890.5 | `e064a9eb5460` |
| `n15_ranked_terminal_centralised_u2000_seed1` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 1 | 3524 | 898.6 | `dd04e571ea5f` |
| `n15_ranked_terminal_centralised_u2000_seed10` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 10 | 3524 | 968.2 | `5c1f14cda46f` |
| `n15_ranked_terminal_centralised_u2000_seed11` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 11 | 3524 | 970.4 | `52ea7b26b396` |
| `n15_ranked_terminal_centralised_u2000_seed12` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 12 | 3524 | 984.7 | `ba7c8d0cb325` |
| `n15_ranked_terminal_centralised_u2000_seed13` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 13 | 3524 | 969.6 | `3dc890132097` |
| `n15_ranked_terminal_centralised_u2000_seed14` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 14 | 3524 | 976.5 | `69af61735902` |
| `n15_ranked_terminal_centralised_u2000_seed15` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 15 | 3524 | 990.2 | `150eca3b3207` |
| `n15_ranked_terminal_centralised_u2000_seed16` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 16 | 3524 | 984.7 | `a6285e186e5d` |
| `n15_ranked_terminal_centralised_u2000_seed17` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 17 | 3524 | 988.4 | `4ba5a4f25c13` |
| `n15_ranked_terminal_centralised_u2000_seed18` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 18 | 3524 | 995.7 | `c1d5f3837271` |
| `n15_ranked_terminal_centralised_u2000_seed19` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 19 | 3524 | 1032.0 | `fb092456f004` |
| `n15_ranked_terminal_centralised_u2000_seed2` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 2 | 3524 | 873.2 | `faf1813f09e9` |
| `n15_ranked_terminal_centralised_u2000_seed3` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 3 | 3524 | 885.0 | `0d5cd3dafa11` |
| `n15_ranked_terminal_centralised_u2000_seed4` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 4 | 3524 | 938.4 | `bd2e1c73ac04` |
| `n15_ranked_terminal_centralised_u2000_seed5` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 5 | 3524 | 883.2 | `82583b3fb0f4` |
| `n15_ranked_terminal_centralised_u2000_seed6` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 6 | 3524 | 882.8 | `e1fa19b581f1` |
| `n15_ranked_terminal_centralised_u2000_seed7` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 7 | 3524 | 888.4 | `37af87e87a02` |
| `n15_ranked_terminal_centralised_u2000_seed8` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 8 | 3524 | 941.5 | `18f6a9ee59f9` |
| `n15_ranked_terminal_centralised_u2000_seed9` | 15 | ranked | terminal | centralised, 45 inputs | 192000 | 9 | 3524 | 891.5 | `40c18ec684cf` |
| `n7_ranked_terminal_centralised_u2000_seed0` | 7 | ranked | terminal | centralised, 21 inputs | 192000 | 0 | 1988 | 413.5 | `8a06afdc50fc` |
| `n7_ranked_terminal_centralised_u2000_seed1` | 7 | ranked | terminal | centralised, 21 inputs | 192000 | 1 | 1988 | 412.3 | `bf2cffe70a69` |
| `n7_ranked_terminal_centralised_u2000_seed10` | 7 | ranked | terminal | centralised, 21 inputs | 192000 | 10 | 1988 | 404.5 | `3df1221e9d42` |
| `n7_ranked_terminal_centralised_u2000_seed11` | 7 | ranked | terminal | centralised, 21 inputs | 192000 | 11 | 1988 | 415.4 | `92bd7a521943` |
| `n7_ranked_terminal_centralised_u2000_seed12` | 7 | ranked | terminal | centralised, 21 inputs | 192000 | 12 | 1988 | 424.9 | `68aeba20d3b4` |
| `n7_ranked_terminal_centralised_u2000_seed13` | 7 | ranked | terminal | centralised, 21 inputs | 192000 | 13 | 1988 | 405.7 | `0c1f9f6e5d96` |
| `n7_ranked_terminal_centralised_u2000_seed14` | 7 | ranked | terminal | centralised, 21 inputs | 192000 | 14 | 1988 | 408.0 | `17d122ceb49a` |
| `n7_ranked_terminal_centralised_u2000_seed15` | 7 | ranked | terminal | centralised, 21 inputs | 192000 | 15 | 1988 | 410.3 | `f3e630648077` |
| `n7_ranked_terminal_centralised_u2000_seed16` | 7 | ranked | terminal | centralised, 21 inputs | 192000 | 16 | 1988 | 412.0 | `ca38873d2f5c` |
| `n7_ranked_terminal_centralised_u2000_seed17` | 7 | ranked | terminal | centralised, 21 inputs | 192000 | 17 | 1988 | 409.6 | `d432f50ad548` |
| `n7_ranked_terminal_centralised_u2000_seed18` | 7 | ranked | terminal | centralised, 21 inputs | 192000 | 18 | 1988 | 403.4 | `6afc2579a38f` |
| `n7_ranked_terminal_centralised_u2000_seed19` | 7 | ranked | terminal | centralised, 21 inputs | 192000 | 19 | 1988 | 278.8 | `19a379e96808` |
| `n7_ranked_terminal_centralised_u2000_seed2` | 7 | ranked | terminal | centralised, 21 inputs | 192000 | 2 | 1988 | 410.6 | `b1377c587144` |
| `n7_ranked_terminal_centralised_u2000_seed3` | 7 | ranked | terminal | centralised, 21 inputs | 192000 | 3 | 1988 | 414.4 | `2341f2d18c17` |
| `n7_ranked_terminal_centralised_u2000_seed4` | 7 | ranked | terminal | centralised, 21 inputs | 192000 | 4 | 1988 | 419.0 | `00dbb8404da6` |
| `n7_ranked_terminal_centralised_u2000_seed5` | 7 | ranked | terminal | centralised, 21 inputs | 192000 | 5 | 1988 | 423.0 | `57631944a83d` |
| `n7_ranked_terminal_centralised_u2000_seed6` | 7 | ranked | terminal | centralised, 21 inputs | 192000 | 6 | 1988 | 434.7 | `223db3950afb` |
| `n7_ranked_terminal_centralised_u2000_seed7` | 7 | ranked | terminal | centralised, 21 inputs | 192000 | 7 | 1988 | 429.6 | `e5f594493935` |
| `n7_ranked_terminal_centralised_u2000_seed8` | 7 | ranked | terminal | centralised, 21 inputs | 192000 | 8 | 1988 | 418.7 | `7a03f600bc95` |
| `n7_ranked_terminal_centralised_u2000_seed9` | 7 | ranked | terminal | centralised, 21 inputs | 192000 | 9 | 1988 | 424.6 | `b95472196a37` |
