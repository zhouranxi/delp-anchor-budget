## A1 — own-segment specificity (bias on own vs random vs all vs rescale)

| dataset | arm | metric | n | Δ | t(9) | p_Holm | d_z | 95% CI | equiv(±0.5) |
|---|---|---|---|---|---|---|---|---|---|
| fashion | random | bwt | 10 | +0.20 | +0.4 | 1.000 | +0.11 | [-1.08, +1.48] | no |
| fashion | repaired | bwt | 10 | -0.20 | -0.3 | 1.000 | -0.08 | [-1.89, +1.49] | no |
| fashion | rescale | bwt | 10 | +0.84 | +0.8 | 1.000 | +0.26 | [-1.45, +3.13] | no |
| fashion | uniform | bwt | 10 | -0.16 | -0.2 | 1.000 | -0.08 | [-1.65, +1.32] | no |
| fashion | random | ell | 10 | -0.21 | -0.9 | 0.406 | -0.28 | [-0.74, +0.33] | no |
| fashion | repaired | ell | 10 | +0.93 | +1.5 | 0.316 | +0.49 | [-0.44, +2.29] | no |
| fashion | rescale | ell | 10 | +4.68 | +10.1 | 0.000 | +3.18 | [+3.63, +5.73] | no |
| fashion | uniform | ell | 10 | +1.40 | +2.8 | 0.065 | +0.88 | [+0.26, +2.54] | no |
| fashion | random | final_avg | 10 | -0.02 | -0.0 | 0.966 | -0.01 | [-1.00, +0.96] | no |
| fashion | repaired | final_avg | 10 | +0.75 | +1.5 | 0.320 | +0.48 | [-0.36, +1.87] | no |
| fashion | rescale | final_avg | 10 | +5.43 | +6.9 | 0.000 | +2.19 | [+3.66, +7.21] | no |
| fashion | uniform | final_avg | 10 | +1.25 | +3.0 | 0.045 | +0.95 | [+0.31, +2.20] | no |
| mnist | random | bwt | 10 | +0.44 | +4.7 | 0.001 | +1.48 | [+0.23, +0.65] | no |
| mnist | repaired | bwt | 10 | +1.76 | +8.9 | 0.000 | +2.82 | [+1.31, +2.21] | no |
| mnist | rescale | bwt | 10 | +3.29 | +16.0 | 0.000 | +5.05 | [+2.82, +3.75] | no |
| mnist | uniform | bwt | 10 | +1.62 | +7.0 | 0.000 | +2.22 | [+1.10, +2.14] | no |
| mnist | random | ell | 10 | -0.19 | -1.0 | 0.336 | -0.32 | [-0.60, +0.23] | no |
| mnist | repaired | ell | 10 | +3.14 | +13.1 | 0.000 | +4.14 | [+2.60, +3.68] | no |
| mnist | rescale | ell | 10 | +6.85 | +24.5 | 0.000 | +7.74 | [+6.21, +7.48] | no |
| mnist | uniform | ell | 10 | +3.00 | +11.2 | 0.000 | +3.54 | [+2.39, +3.60] | no |
| mnist | random | final_avg | 10 | +0.21 | +1.0 | 0.354 | +0.31 | [-0.28, +0.70] | no |
| mnist | repaired | final_avg | 10 | +4.72 | +18.1 | 0.000 | +5.71 | [+4.13, +5.31] | no |
| mnist | rescale | final_avg | 10 | +9.81 | +32.7 | 0.000 | +10.35 | [+9.13, +10.48] | no |
| mnist | uniform | final_avg | 10 | +4.46 | +13.7 | 0.000 | +4.35 | [+3.72, +5.19] | no |

### Each arm relative to the *repaired* arm (own-segment bias)

| dataset | arm | metric | Δ (arm − repaired) | t | p_Holm | 90% CI |
|---|---|---|---|---|---|---|
| fashion | base | ell | -0.93 | -1.5 | 0.158 | [-2.03, +0.18] |
| fashion | base | final_avg | -0.75 | -1.5 | 0.160 | [-1.66, +0.15] |
| fashion | random | ell | -1.13 | -2.2 | 0.059 | [-2.10, -0.17] |
| fashion | random | final_avg | -0.77 | -1.3 | 0.230 | [-1.87, +0.33] |
| fashion | rescale | ell | +3.75 | +7.2 | 0.000 | [+2.79, +4.70] |
| fashion | rescale | final_avg | +4.68 | +4.8 | 0.001 | [+2.88, +6.48] |
| fashion | uniform | ell | +0.47 | +1.3 | 0.224 | [-0.19, +1.13] |
| fashion | uniform | final_avg | +0.50 | +1.0 | 0.350 | [-0.43, +1.43] |
| mnist | base | ell | -3.14 | -13.1 | 0.000 | [-3.58, -2.70] |
| mnist | base | final_avg | -4.72 | -18.1 | 0.000 | [-5.20, -4.24] |
| mnist | random | ell | -3.32 | -10.7 | 0.000 | [-3.89, -2.75] |
| mnist | random | final_avg | -4.51 | -11.5 | 0.000 | [-5.23, -3.79] |
| mnist | rescale | ell | +3.71 | +19.6 | 0.000 | [+3.36, +4.06] |
| mnist | rescale | final_avg | +5.08 | +18.6 | 0.000 | [+4.58, +5.59] |
| mnist | uniform | ell | -0.14 | -1.3 | 0.238 | [-0.34, +0.06] |
| mnist | uniform | final_avg | -0.26 | -1.3 | 0.227 | [-0.64, +0.11] |

## A2 — dominant-segment instrument positive control (oracle)

| dataset | arm | n | own_rate (last phase) | 95% CI | n_active | ell | avg |
|---|---|---|---|---|---|---|---|
| fashion | base | 10 | 0.0568 | [0.002, 0.181] | 97 | 56.83 | 42.40 |
| fashion | oracle | 10 | 1.0000 | [0.964, 1.000] | 106 | 58.00 | 43.90 |
| fashion | weak_oracle | 10 | 1.0000 | [0.961, 1.000] | 99 | 56.92 | 42.66 |
| mnist | base | 10 | 0.0551 | [0.005, 0.188] | 107 | 74.38 | 63.83 |
| mnist | oracle | 10 | 1.0000 | [0.967, 1.000] | 119 | 77.33 | 68.19 |
| mnist | weak_oracle | 10 | 1.0000 | [0.965, 1.000] | 113 | 75.98 | 66.18 |

### Oracle arms vs base (performance)

| dataset | arm | metric | n | Δ | t | p_Holm | d_z | 95% CI |
|---|---|---|---|---|---|---|---|---|
| fashion | oracle | bwt | 10 | +0.35 | +0.4 | 1.000 | +0.13 | [-1.57, +2.28] |
| fashion | weak_oracle | bwt | 10 | +0.17 | +0.3 | 1.000 | +0.09 | [-1.21, +1.55] |
| fashion | oracle | ell | 10 | +1.17 | +2.5 | 0.069 | +0.79 | [+0.11, +2.23] |
| fashion | weak_oracle | ell | 10 | +0.10 | +0.3 | 0.804 | +0.08 | [-0.75, +0.94] |
| fashion | oracle | final_avg | 10 | +1.49 | +2.0 | 0.154 | +0.63 | [-0.20, +3.19] |
| fashion | weak_oracle | final_avg | 10 | +0.25 | +0.8 | 0.442 | +0.25 | [-0.46, +0.96] |
| mnist | oracle | bwt | 10 | +1.57 | +10.0 | 0.000 | +3.16 | [+1.21, +1.92] |
| mnist | weak_oracle | bwt | 10 | +0.83 | +7.8 | 0.000 | +2.48 | [+0.59, +1.08] |
| mnist | oracle | ell | 10 | +2.95 | +11.5 | 0.000 | +3.63 | [+2.37, +3.54] |
| mnist | weak_oracle | ell | 10 | +1.60 | +7.0 | 0.000 | +2.22 | [+1.08, +2.12] |
| mnist | oracle | final_avg | 10 | +4.37 | +13.9 | 0.000 | +4.41 | [+3.66, +5.08] |
| mnist | weak_oracle | final_avg | 10 | +2.35 | +11.1 | 0.000 | +3.51 | [+1.87, +2.83] |

## A3 — segment-resolved anchor construction variants

| dataset | variant | metric | n | ell | avg | Δ vs neuron | t | p_Holm | 90% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| fashion | branch_v1 | Δell | 10 | 56.79 | 41.68 | -0.04 | -0.2 | 0.867 | [-0.51, +0.43] |
| fashion | branch_v2 | Δell | 10 | 58.65 | 32.96 | +1.82 | +4.6 | 0.001 | [+1.09, +2.55] |
| fashion | branch_v3 | Δell | 10 | 58.65 | 32.96 | +1.82 | +4.6 | 0.001 | [+1.09, +2.55] |
| fashion | branch_v1 | Δavg | 10 | 56.79 | 41.68 | -0.73 | -1.3 | 0.234 | [-1.77, +0.32] |
| fashion | branch_v2 | Δavg | 10 | 58.65 | 32.96 | -9.45 | -17.4 | 0.000 | [-10.44, -8.46] |
| fashion | branch_v3 | Δavg | 10 | 58.65 | 32.96 | -9.45 | -17.4 | 0.000 | [-10.44, -8.46] |
| mnist | branch_v1 | Δell | 10 | 74.42 | 64.07 | +0.05 | +0.3 | 0.757 | [-0.22, +0.31] |
| mnist | branch_v2 | Δell | 10 | 76.87 | 44.10 | +2.49 | +14.3 | 0.000 | [+2.17, +2.81] |
| mnist | branch_v3 | Δell | 10 | 76.87 | 44.10 | +2.49 | +14.3 | 0.000 | [+2.17, +2.81] |
| mnist | branch_v1 | Δavg | 10 | 74.42 | 64.07 | +0.25 | +1.4 | 0.193 | [-0.07, +0.57] |
| mnist | branch_v2 | Δavg | 10 | 76.87 | 44.10 | -19.72 | -25.8 | 0.000 | [-21.12, -18.32] |
| mnist | branch_v3 | Δavg | 10 | 76.87 | 44.10 | -19.72 | -25.8 | 0.000 | [-21.12, -18.32] |

