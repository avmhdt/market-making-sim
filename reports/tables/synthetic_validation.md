| check | case | metric | value | target | SE | z | tolerance | pass |
|---|---|---|---|---|---|---|---|---|
| AC-W-01 | delta=0.1 | fill_rate_per_s | 120.552 | 120.499 |  |  | 99% Poisson CI [119.555, 121.556] | yes |
| AC-W-01 | delta=0.2 | fill_rate_per_s | 103.616 | 103.715 |  |  | 99% Poisson CI [102.692, 104.547] | yes |
| AC-W-01 | delta=0.3 | fill_rate_per_s | 89.0962 | 89.2679 |  |  | 99% Poisson CI [88.239, 89.959] | yes |
| AC-W-01 | delta=0.4 | fill_rate_per_s | 76.8912 | 76.8336 |  |  | 99% Poisson CI [76.095, 77.693] | yes |
| AC-W-01 | delta=0.5 | fill_rate_per_s | 66.1612 | 66.1313 |  |  | 99% Poisson CI [65.423, 66.906] | yes |
| AC-W-01 | delta=0.6 | fill_rate_per_s | 57.0262 | 56.9198 |  |  | 99% Poisson CI [56.341, 57.718] | yes |
| AC-W-01 | delta=0.7 | fill_rate_per_s | 49.0562 | 48.9913 |  |  | 99% Poisson CI [48.421, 49.698] | yes |
| AC-W-01 | delta=0.8 | fill_rate_per_s | 42.2338 | 42.1672 |  |  | 99% Poisson CI [41.644, 42.829] | yes |
| AC-W-01 | delta=0.9 | fill_rate_per_s | 36.4113 | 36.2936 |  |  | 99% Poisson CI [35.864, 36.964] | yes |
| AC-W-01 | delta=1.0 | fill_rate_per_s | 31.3513 | 31.2382 |  |  | 99% Poisson CI [30.844, 31.865] | yes |
| AC-W-01 | fit | k | 1.49583 | 1.5 |  |  | within 3% | yes |
| AC-W-01 | fit | A | 139.819 | 140 |  |  | reported | yes |
| AC-W-02 | gamma=0.1 inventory reference | mean_profit | 64.9328 | 65 | 0.2064 | -0.33 | |z| <= 3 | yes |
| AC-W-02 | gamma=0.1 inventory reference | std_profit | 6.52724 | 6.6 | 0.146 | -0.50 | |z| <= 3 | yes |
| AC-W-02 | gamma=0.1 inventory reference | mean_final_q | -0.055 | 0.08 | 0.09135 | -1.48 | |z| <= 3 | yes |
| AC-W-02 | gamma=0.1 inventory reference | std_final_q | 2.88865 | 2.9 | 0.06459 | -0.18 | |z| <= 3 | yes |
| AC-W-02 | gamma=0.1 inventory reference | avg_spread | 1.49177 | 1.49 |  |  | reported | yes |
| AC-W-02 diagnostic | gamma=0.1 inventory reference 50k | mean_profit | 64.8759 | 65 | 0.206 | 0.60 | reported, post hoc: z of the published value | yes |
| AC-W-02 diagnostic | gamma=0.1 inventory reference 50k | std_profit | 6.51564 | 6.6 | 0.1457 | 0.58 | reported, post hoc: z of the published value | yes |
| AC-W-02 diagnostic | gamma=0.1 inventory reference 50k | mean_final_q | -0.0069 | 0.08 | 0.09264 | 0.94 | reported, post hoc: z of the published value | yes |
| AC-W-02 diagnostic | gamma=0.1 inventory reference 50k | std_final_q | 2.92939 | 2.9 | 0.0655 | -0.45 | reported, post hoc: z of the published value | yes |
| AC-W-02 | gamma=0.1 inventory fallback | mean_profit | 64.8486 | 65 | 0.2099 | -0.72 | |z| <= 3 | yes |
| AC-W-02 | gamma=0.1 inventory fallback | std_profit | 6.63634 | 6.6 | 0.1484 | 0.24 | |z| <= 3 | yes |
| AC-W-02 | gamma=0.1 inventory fallback | mean_final_q | 0.1 | 0.08 | 0.09451 | 0.21 | |z| <= 3 | yes |
| AC-W-02 | gamma=0.1 inventory fallback | std_final_q | 2.9888 | 2.9 | 0.06683 | 1.33 | |z| <= 3 | yes |
| AC-W-04 | gamma=0.1 inventory | fallback_minus_reference_profit | -0.0842549 | 0 | 0.2944 | -0.29 | |z| <= 3 | yes |
| AC-W-02 | gamma=0.1 symmetric reference | mean_profit | 68.1739 | 68.4 | 0.4513 | -0.50 | |z| <= 3 | yes |
| AC-W-02 | gamma=0.1 symmetric reference | std_profit | 14.2729 | 12.7 | 0.3192 | 4.93 | |z| <= 3 | NO, documented exception (M2-20) |
| AC-W-02 | gamma=0.1 symmetric reference | mean_final_q | -0.129 | 0.26 | 0.2714 | -1.43 | |z| <= 3 | yes |
| AC-W-02 | gamma=0.1 symmetric reference | std_final_q | 8.58173 | 8.4 | 0.1919 | 0.95 | |z| <= 3 | yes |
| AC-W-02 | gamma=0.1 symmetric reference | avg_spread | 1.49177 | 1.49 |  |  | reported | yes |
| AC-W-02 diagnostic | gamma=0.1 symmetric reference 50k | mean_profit | 68.2991 | 68.4 | 0.4281 | 0.24 | reported, post hoc: z of the published value | yes |
| AC-W-02 diagnostic | gamma=0.1 symmetric reference 50k | std_profit | 13.5374 | 12.7 | 0.3027 | -2.77 | reported, post hoc: z of the published value | yes |
| AC-W-02 diagnostic | gamma=0.1 symmetric reference 50k | mean_final_q | 0.06136 | 0.26 | 0.2652 | 0.75 | reported, post hoc: z of the published value | yes |
| AC-W-02 diagnostic | gamma=0.1 symmetric reference 50k | std_final_q | 8.38701 | 8.4 | 0.1875 | 0.07 | reported, post hoc: z of the published value | yes |
| AC-W-02 | gamma=0.1 symmetric fallback | mean_profit | 67.9852 | 68.4 | 0.4027 | -1.03 | |z| <= 3 | yes |
| AC-W-02 | gamma=0.1 symmetric fallback | std_profit | 12.7332 | 12.7 | 0.2847 | 0.12 | |z| <= 3 | yes |
| AC-W-02 | gamma=0.1 symmetric fallback | mean_final_q | 0.317 | 0.26 | 0.2667 | 0.21 | |z| <= 3 | yes |
| AC-W-02 | gamma=0.1 symmetric fallback | std_final_q | 8.43503 | 8.4 | 0.1886 | 0.19 | |z| <= 3 | yes |
| AC-W-02 | gamma=0.01 inventory reference | mean_profit | 68.4406 | 68.6 | 0.2828 | -0.56 | |z| <= 3 | yes |
| AC-W-02 | gamma=0.01 inventory reference | std_profit | 8.94363 | 8.7 | 0.2 | 1.22 | |z| <= 3 | yes |
| AC-W-02 | gamma=0.01 inventory reference | mean_final_q | -0.105 | 0.12 | 0.1675 | -1.34 | |z| <= 3 | yes |
| AC-W-02 | gamma=0.01 inventory reference | std_final_q | 5.29679 | 5.1 | 0.1184 | 1.66 | |z| <= 3 | yes |
| AC-W-02 | gamma=0.01 inventory reference | avg_spread | 1.34901 | 1.35 |  |  | reported | yes |
| AC-W-02 diagnostic | gamma=0.01 inventory reference 50k | mean_profit | 68.392 | 68.6 | 0.2839 | 0.73 | reported, post hoc: z of the published value | yes |
| AC-W-02 diagnostic | gamma=0.01 inventory reference 50k | std_profit | 8.97894 | 8.7 | 0.2008 | -1.39 | reported, post hoc: z of the published value | yes |
| AC-W-02 diagnostic | gamma=0.01 inventory reference 50k | mean_final_q | -0.02794 | 0.12 | 0.1653 | 0.90 | reported, post hoc: z of the published value | yes |
| AC-W-02 diagnostic | gamma=0.01 inventory reference 50k | std_final_q | 5.22614 | 5.1 | 0.1169 | -1.08 | reported, post hoc: z of the published value | yes |
| AC-W-02 | gamma=0.01 inventory fallback | mean_profit | 68.2672 | 68.6 | 0.2701 | -1.23 | |z| <= 3 | yes |
| AC-W-02 | gamma=0.01 inventory fallback | std_profit | 8.5408 | 8.7 | 0.191 | -0.83 | |z| <= 3 | yes |
| AC-W-02 | gamma=0.01 inventory fallback | mean_final_q | 0.1 | 0.12 | 0.168 | -0.12 | |z| <= 3 | yes |
| AC-W-02 | gamma=0.01 inventory fallback | std_final_q | 5.31208 | 5.1 | 0.1188 | 1.79 | |z| <= 3 | yes |
| AC-W-02 | gamma=0.01 symmetric reference | mean_profit | 68.1106 | 68.8 | 0.4463 | -1.54 | |z| <= 3 | yes |
| AC-W-02 | gamma=0.01 symmetric reference | std_profit | 14.1133 | 12.8 | 0.3156 | 4.16 | |z| <= 3 | NO, documented exception (M2-20) |
| AC-W-02 | gamma=0.01 symmetric reference | mean_final_q | 0.163 | 0.09 | 0.2896 | 0.25 | |z| <= 3 | yes |
| AC-W-02 | gamma=0.01 symmetric reference | std_final_q | 9.15742 | 8.7 | 0.2048 | 2.23 | |z| <= 3 | yes |
| AC-W-02 | gamma=0.01 symmetric reference | avg_spread | 1.34901 | 1.35 |  |  | reported | yes |
| AC-W-02 diagnostic | gamma=0.01 symmetric reference 50k | mean_profit | 68.6973 | 68.8 | 0.4323 | 0.24 | reported, post hoc: z of the published value | yes |
| AC-W-02 diagnostic | gamma=0.01 symmetric reference 50k | std_profit | 13.6713 | 12.8 | 0.3057 | -2.85 | reported, post hoc: z of the published value | yes |
| AC-W-02 diagnostic | gamma=0.01 symmetric reference 50k | mean_final_q | -0.00192 | 0.09 | 0.2752 | 0.33 | reported, post hoc: z of the published value | yes |
| AC-W-02 diagnostic | gamma=0.01 symmetric reference 50k | std_final_q | 8.70241 | 8.7 | 0.1946 | -0.01 | reported, post hoc: z of the published value | yes |
| AC-W-02 | gamma=0.01 symmetric fallback | mean_profit | 68.5286 | 68.8 | 0.4102 | -0.66 | |z| <= 3 | yes |
| AC-W-02 | gamma=0.01 symmetric fallback | std_profit | 12.9713 | 12.8 | 0.29 | 0.59 | |z| <= 3 | yes |
| AC-W-02 | gamma=0.01 symmetric fallback | mean_final_q | 0.215 | 0.09 | 0.2834 | 0.44 | |z| <= 3 | yes |
| AC-W-02 | gamma=0.01 symmetric fallback | std_final_q | 8.96109 | 8.7 | 0.2004 | 1.30 | |z| <= 3 | yes |
| AC-W-02 | gamma=1 inventory reference | mean_profit | 31.38 | 31.4 | 0.1546 | -0.13 | |z| <= 3 | yes |
| AC-W-02 | gamma=1 inventory reference | std_profit | 4.888 | 5 | 0.1093 | -1.02 | |z| <= 3 | yes |
| AC-W-02 | gamma=1 inventory reference | mean_final_q | -0.028 | 0.02 | 0.05255 | -0.91 | |z| <= 3 | yes |
| AC-W-02 | gamma=1 inventory reference | std_final_q | 1.66192 | 1.7 | 0.03716 | -1.02 | |z| <= 3 | yes |
| AC-W-02 | gamma=1 inventory reference | avg_spread | 3.03165 | 3.02 |  |  | reported | yes |
| AC-W-02 diagnostic | gamma=1 inventory reference 50k | mean_profit | 31.3981 | 31.4 | 0.1532 | 0.01 | reported, post hoc: z of the published value | yes |
| AC-W-02 diagnostic | gamma=1 inventory reference 50k | std_profit | 4.84526 | 5 | 0.1083 | 1.43 | reported, post hoc: z of the published value | yes |
| AC-W-02 diagnostic | gamma=1 inventory reference 50k | mean_final_q | 0.0053 | 0.02 | 0.05212 | 0.28 | reported, post hoc: z of the published value | yes |
| AC-W-02 diagnostic | gamma=1 inventory reference 50k | std_final_q | 1.64833 | 1.7 | 0.03686 | 1.40 | reported, post hoc: z of the published value | yes |
| AC-W-02 | gamma=1 inventory fallback | mean_profit | 31.3956 | 31.4 | 0.1579 | -0.03 | |z| <= 3 | yes |
| AC-W-02 | gamma=1 inventory fallback | std_profit | 4.99267 | 5 | 0.1116 | -0.07 | |z| <= 3 | yes |
| AC-W-02 | gamma=1 inventory fallback | mean_final_q | 0.037 | 0.02 | 0.05195 | 0.33 | |z| <= 3 | yes |
| AC-W-02 | gamma=1 inventory fallback | std_final_q | 1.64266 | 1.7 | 0.03673 | -1.56 | |z| <= 3 | yes |
| AC-W-02 | gamma=1 symmetric reference | mean_profit | 44.0351 | 44 | 0.336 | 0.10 | |z| <= 3 | yes |
| AC-W-02 | gamma=1 symmetric reference | std_profit | 10.6257 | 11 | 0.2376 | -1.58 | |z| <= 3 | yes |
| AC-W-02 | gamma=1 symmetric reference | mean_final_q | -0.106 | 0 | 0.151 | -0.70 | |z| <= 3 | yes |
| AC-W-02 | gamma=1 symmetric reference | std_final_q | 4.77636 | 5.1 | 0.1068 | -3.03 | |z| <= 3 | NO, documented exception (M2-20) |
| AC-W-02 | gamma=1 symmetric reference | avg_spread | 3.03165 | 3.02 |  |  | reported | yes |
| AC-W-02 diagnostic | gamma=1 symmetric reference 50k | mean_profit | 43.7198 | 44 | 0.3382 | 0.83 | reported, post hoc: z of the published value | yes |
| AC-W-02 diagnostic | gamma=1 symmetric reference 50k | std_profit | 10.6955 | 11 | 0.2392 | 1.27 | reported, post hoc: z of the published value | yes |
| AC-W-02 diagnostic | gamma=1 symmetric reference 50k | mean_final_q | -0.03202 | 0 | 0.1637 | 0.20 | reported, post hoc: z of the published value | yes |
| AC-W-02 diagnostic | gamma=1 symmetric reference 50k | std_final_q | 5.17512 | 5.1 | 0.1157 | -0.65 | reported, post hoc: z of the published value | yes |
| AC-W-02 | gamma=1 symmetric fallback | mean_profit | 43.6079 | 44 | 0.3476 | -1.13 | |z| <= 3 | yes |
| AC-W-02 | gamma=1 symmetric fallback | std_profit | 10.9913 | 11 | 0.2458 | -0.04 | |z| <= 3 | yes |
| AC-W-02 | gamma=1 symmetric fallback | mean_final_q | 0.057 | 0 | 0.1645 | 0.35 | |z| <= 3 | yes |
| AC-W-02 | gamma=1 symmetric fallback | std_final_q | 5.202 | 5.1 | 0.1163 | 0.88 | |z| <= 3 | yes |
| AC-W-04 | conservative vs front | fills | 1946 | 2357 |  |  | strictly lower | yes |
| AC-W-04 | +50 ms round trip vs 0 | fills | 2287 | 2357 |  |  | strictly lower | yes |
| AC-W-03 | gamma=0.1 sigma=2 A=140 k=1.5 Q=5 | expected_utility | -0.00174026 | -0.00174582 | 1.068e-05 | 0.52 | <= 3 SE + 1% | yes |
| AC-W-03 | gamma=0.1 sigma=2 A=140 k=1.5 Q=5 | certainty_equivalent | 63.5372 | 63.5053 |  |  | reported | yes |
| AC-W-03 | gamma=0.5 sigma=2 A=140 k=1.5 Q=5 | expected_utility | -1.87781e-12 | -2.00634e-12 | 3.906e-13 | 0.33 | <= 3 SE + 1% | yes |
| AC-W-03 | gamma=0.5 sigma=2 A=140 k=1.5 Q=5 | certainty_equivalent | 54.0018 | 53.8694 |  |  | reported | yes |
| AC-W-03 | gamma=0.05 sigma=2 A=140 k=1.5 Q=10 | expected_utility | -0.0368937 | -0.0370286 | 0.0001203 | 1.12 | <= 3 SE + 1% | yes |
| AC-W-03 | gamma=0.05 sigma=2 A=140 k=1.5 Q=10 | certainty_equivalent | 65.9943 | 65.9213 |  |  | reported | yes |
| AC-W-03 | gamma=0.2 sigma=1 A=50 k=1 Q=3 | expected_utility | -0.00201269 | -0.00204089 | 2.054e-05 | 1.37 | <= 3 SE + 1% | yes |
| AC-W-03 | gamma=0.2 sigma=1 A=50 k=1 Q=3 | certainty_equivalent | 31.0414 | 30.9719 |  |  | reported | yes |
| AC-W-14 | gamma=0.01 sigma=0.3 A=0.9 k=0.3 Q=30 | max_rel_gap_inner | 0.000552356 | 0.05 |  |  | <= 5% for |q| <= Q/2 | yes |
| AC-W-14 | gamma=0.01 sigma=0.3 A=0.9 k=0.3 Q=30 | max_rel_gap_near_bounds | 0.160932 | 0 |  |  | reported | yes |
| AC-W-14 | gamma=0.1 sigma=2 A=140 k=1.5 Q=10 | max_rel_gap_inner | 0.00643786 | 0.05 |  |  | <= 5% for |q| <= Q/2 | yes |
| AC-W-14 | gamma=0.1 sigma=2 A=140 k=1.5 Q=10 | max_rel_gap_near_bounds | 0.132112 | 0 |  |  | reported | yes |
| AC-W-14 | gamma=0.05 sigma=0.3 A=0.9 k=0.3 Q=20 | max_rel_gap_inner | 0.00241652 | 0.05 |  |  | <= 5% for |q| <= Q/2 | yes |
| AC-W-14 | gamma=0.05 sigma=0.3 A=0.9 k=0.3 Q=20 | max_rel_gap_near_bounds | 0.10405 | 0 |  |  | reported | yes |
| AC-W-14 | gamma=0.5 sigma=2 A=140 k=1.5 Q=5 | max_rel_gap_inner | 0.0204998 | 0.05 |  |  | <= 5% for |q| <= Q/2 | yes |
| AC-W-14 | gamma=0.5 sigma=2 A=140 k=1.5 Q=5 | max_rel_gap_near_bounds | 0.1853 | 0 |  |  | reported | yes |
| AC-W-14 | all sets | sets_within_5pct | 4 | 3 |  |  | >= 3 sets | yes |
