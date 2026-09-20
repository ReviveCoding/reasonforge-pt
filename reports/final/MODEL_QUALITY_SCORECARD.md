# Model quality / release scorecard

| Dimension | Result |
|---|---|
| Scientific decision | RETAIN_CONVENTIONAL_4B (unchanged) |
| Frozen primary C2-C1 | +0.008519 |
| Frozen CP100 95% interval | [-0.006296, +0.025556] |
| Promotion threshold | +0.030000; failed |
| Crossed post-final sensitivity | [-0.010370, +0.030741] (does not replace CP100) |
| Repeated-example cells | 450 units = 150 examples x 3 seeds |
| Binary recoveries / regressions | 11 / 8; pooled counts descriptive |
| Routing | NO_ROBUST_ROUTING_BENEFIT |
| Uncertainty | UNCERTAINTY_QUALITY_ASSOCIATION; grouped mean Brier 0.2354 |
| Uncertainty proxy coverage | finite log-prob 649/1800; entropy 0/1800 |
| D114 | CASE-B corrected top8/family; 432 diagnostic rows |
| D115 | CASE-B corrected top6/family; 144 rows; timestamp-integrated estimated energy 34.793 Wh |
| Robustness | Bounded post-CP100 diagnostic evidence only |
| Human review | 60 blinded pairs; labels pending |
| Public benchmark contamination | Cannot be excluded |
| Three seeds | Not high-powered population inference |

All post-final analyses are non-promotion-eligible and do not overwrite E47.
