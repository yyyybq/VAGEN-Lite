# r5 production execution graph

Status: `PASS` (reconstructed from the r5 resolved config and current production code).

The r5 configuration resolves to `algorithm.adv_estimator=no_concat_gae`,
`gamma=0.95`, `lam=0.95`, `trainer.balance_batch=true`,
`filter.enable=false`, and `algorithm.use_kl_in_reward=false`.

```text
8 train sources x rollout.n=4
  -> no-concat agent loops emit one row per environment turn
  -> _post_process_no_concat_batch: 297 rows
  -> compute response_mask
  -> pad_dataproto_to_divisor(4): prefix-copy 3 rows, 297 -> 300
  -> _balance_batch: reorder all 300 rows for DP token balance
  -> reward function produces reward_tensor
  -> actor old logprob
  -> reference logprob
  -> critic values
  -> token_level_scores = reward_tensor
  -> token_level_rewards = token_level_scores (KL-in-reward disabled)
  -> production no_concat_gae
       -> normalize IDs: group factorized, traj/turn cast to int64
       -> unique by (group, traj, turn): 300 -> 297 semantic turns
       -> sum token reward per turn
       -> extract critic value at FIRST valid response token
       -> sort each (group, traj) trajectory by numeric turn_idx
       -> historical reverse GAE, trajectory-end next_value=0
       -> whiten on 297 unique rows x 17,801 valid response tokens
       -> broadcast turn advantage to valid response tokens
       -> expand unique results through inverse map: 297 -> 300 rows
       -> return target written only at first valid response token
  -> compute_value_mask from return sentinel
  -> filter skipped (disabled)
  -> snapshot critic initialization identity
  -> exact snapshot of 300-row post-GAE batch
  -> return before update_critic/update_actor
```

## Code evidence

- No-concat flattening: `vagen/ray_trainer.py:2491-2498`.
- Padding and DP reorder occur before reward/value/GAE:
  `vagen/ray_trainer.py:2500-2513`.
- Reward creation: `vagen/ray_trainer.py:2521-2532`.
- Actor/ref logprob and critic values: `vagen/ray_trainer.py:2549-2611`.
- `token_level_scores` assignment: `vagen/ray_trainer.py:2613-2618`.
- KL-in-reward disabled path copies scores to rewards:
  `vagen/ray_trainer.py:2739-2746`.
- Advantage dispatch: `vagen/ray_trainer.py:2765-2778`.
- Value mask creation: `vagen/ray_trainer.py:2780-2781`.
- Filter is evaluated after GAE and is disabled in r5:
  `vagen/ray_trainer.py:2790-2799`.
- Snapshot boundary and stop-before-update:
  `vagen/ray_trainer.py:2801-2850`.
- Production ID normalization and unique-row construction:
  `vagen/custom_advantage/no_concat_gae.py:216-254`.
- First-token value extraction:
  `vagen/custom_advantage/no_concat_gae.py:262-278`.
- Numeric trajectory sorting and historical recursion:
  `vagen/custom_advantage/no_concat_gae.py:280-324`.
- Unique-population whitening and duplicate expansion:
  `vagen/custom_advantage/no_concat_gae.py:326-336`.

The proposed hypothesis “GAE/whitening on 297 rows, then padding to 300” is
therefore false for r5. Padding happens first; the production estimator then
removes the three duplicate triples for GAE and whitening and expands the
results back afterward.
