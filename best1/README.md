# best1: CMT/ALIGNN N2 best single run archive

This folder preserves the strongest single N2 finetuning result observed so far.

## Result

- Downstream: `N2`
- Seed: `0`
- Pretrain checkpoint: aggressive CMT version 2
- Finetune checkpoint: epoch 77, step 7878
- Test MAE: `0.020818566903471947`
- Test loss: `0.0009826116729527712`
- Test R2: `0.8044009208679199`
- Val MAE: `0.022936642169952393`
- Val loss: `0.001285927719436586`

## Archived Files

- `checkpoints/pretrain_cmt_aggressive_version2_best.ckpt`
- `checkpoints/finetune_n2_seed0_epoch77_step7878_best.ckpt`
- `logs/pretrain_alignn_hmof_1pct_cmt_aggressive_20260406_045354.log`
- `logs/try_alignn_with_pretrain_20260414_015240.log`
- `scripts/run_pretrain_alignn_hmof_1pct_cmt_aggressive.sh`
- `scripts/run_try_alignn_with_pretrain.sh`

## Important Caveat

This is a real result, but current multi-seed reruns show high variance. Treat it as
the best single run, not as the stable average performance of the current method.

Observed seed-sensitive reruns:

- Seed 1, original reproduction: test MAE around `0.03223`
- Seed 2, original reproduction: test MAE around `0.02815`
- Seed 1, grouped LR attempt: test MAE around `0.02524`
- Seed 1, zero-initialized regression head: test MAE around `0.03405`

The next stable-improvement direction is to normalize downstream regression labels
with train-set statistics and then rerun seed 0/1/2 under the same protocol.
