# Pretraining Ablation Plan

Goal: choose task groups before tuning weights.

## Stage 1: cheap screening

Run these 4 pretraining profiles:

1. `mpp_only`
2. `mpp_chem_basic`
3. `mpp_zeopp_basic`
4. `mpp_chem_basic_zeopp_basic`

All four use:

- short pretraining (`max_epochs=10` by default)
- no uncertainty weighting by default
- same architecture as current CMT/ALIGNN line

## Labels used

### chem_basic

- `density`
- `metal_fraction`
- `fraction_C`
- `fraction_N`
- `fraction_O`

### zeopp_basic

- `pld`
- `lcd`
- `av_fraction`
- `av_cm3_g`

## Stage 2: quick downstream evaluation

For each candidate pretrained ckpt, evaluate with the best1 finetuning line:

- fixed downstream recipe
- `max_epochs=50` quick screen
- compare primarily on `val MAE`, secondarily `test MAE` / `R2`

## Decision rule

1. If `mpp + zeopp_basic` is clearly best, prioritize zeopp and drop chem.
2. If `mpp + chem_basic` is clearly best, prioritize chem and drop zeopp.
3. If `mpp + chem_basic + zeopp_basic` is best, keep both and move on to weight tuning.
4. If `mpp only` is already best, auxiliary descriptors are hurting and should be simplified before any weight tuning.
