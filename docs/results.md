# Results

The frozen 4B primary result was C2 − C1 = **+0.008519**, with a seed-stratified 95% interval of **[-0.006296, +0.025556]**. The preregistered practical threshold was **+0.030000**. The gate failed and the decision remains **RETAIN_CONVENTIONAL_4B**. Small positive observed seed differences do not establish RCAC superiority.

| Seed | C1 score | C2 score | C2 − C1 |
|---|---:|---:|---:|
| 1701 | 0.369167 | 0.382500 | +0.013333 |
| 1702 | 0.379167 | 0.385833 | +0.006667 |
| 1703 | 0.385833 | 0.391389 | +0.005556 |

The [primary seed table](../reports/tables/primary_seed_effects.csv) is a derived transcription of the frozen E47 decision. The [paired-effect figure](../reports/figures/paired_delta_distribution.svg) provides a visual summary.

The later analytics mart contains 1,452 evaluation units and 558 matched C1/C2 pairs. The locked subset has 450 seed-example cells from 150 examples. Corrected diagnostic inference produced 432 D114R generation-policy rows and 144 D115R systems rows. The D112 stochastic study contains 1,800 rollouts. A 60-pair blinded human-review packet exists, but labels are pending.

The post-final results include 11 pooled recoveries and 8 regressions across repeated cells; those pooled counts are descriptive. Seed-specific McNemar results are separate. Example-group-held-out routing found **NO_ROBUST_ROUTING_BENEFIT**. Uncertainty evidence supports **UNCERTAINTY_QUALITY_ASSOCIATION** only; mean token entropy was unavailable in all 1,800 D112 rows, and finite sequence log probability covered 649 rows. D115R energy is an estimate integrated over actual telemetry timestamps, with no claim of device-active GPU time.

Selected aggregate tables appear in [`reports/tables/`](../reports/tables/) and the [model-quality scorecard](../reports/final/MODEL_QUALITY_SCORECARD.md). These post-final findings do not replace the primary interval or reopen the decision.
