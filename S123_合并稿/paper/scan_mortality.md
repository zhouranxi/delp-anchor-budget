# A5: mortality and repair gain in the 2x2 width x normalization scan

Ten seeds per cell, paired b=10 against b=0 within the same seed.

| dataset | width | GroupNorm | n | mortality (b=0) | mortality (b=10) | Δ final acc | t | p | d_z | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| fashion | 2000 | no | 10 | 0.952 | 0.948 | +2.13 | +3.4 | 0.008 | +1.08 | [+0.91, +3.36] |
| fashion | 2000 | yes | 10 | 0.875 | 0.884 | -6.64 | -4.0 | 0.003 | -1.27 | [-9.90, -3.39] |
| fashion | 256 | no | 10 | 0.618 | 0.592 | +0.90 | +2.1 | 0.063 | +0.67 | [+0.07, +1.74] |
| fashion | 256 | yes | 10 | 0.010 | 0.020 | +0.42 | +1.1 | 0.321 | +0.33 | [-0.37, +1.21] |
| mnist | 2000 | no | 10 | 0.945 | 0.939 | +4.62 | +12.5 | 0.000 | +3.95 | [+3.90, +5.35] |
| mnist | 2000 | yes | 10 | 0.745 | 0.780 | -2.74 | -2.3 | 0.044 | -0.74 | [-5.04, -0.45] |
| mnist | 256 | no | 10 | 0.584 | 0.535 | +4.67 | +15.7 | 0.000 | +4.96 | [+4.09, +5.26] |
| mnist | 256 | yes | 10 | 0.004 | 0.000 | +0.95 | +6.2 | 0.000 | +1.95 | [+0.65, +1.25] |

## Correlation between mortality and repair gain across cells

| subset | cells | Pearson r | p | Spearman rho | p |
|---|---|---|---|---|---|
| fashion | 4 | -0.271 | 0.729 | +0.400 | 0.600 |
| mnist | 4 | +0.110 | 0.890 | +0.000 | 1.000 |
| pooled | 8 | -0.112 | 0.792 | +0.048 | 0.911 |
