# Targeted correlation filtering

Both unscaled Logistic Regression models were fitted on the same pregnancy-wise training split and evaluated on the same validation recordings. The targeted model removed `F_Mean`, `F_BL`, and `U_F`.

| Metric | Baseline | Targeted filter |
|---|---:|---:|
| Features | 103 | 100 |
| Training abnormal F1 | 0.7756 | 0.7756 |
| Validation abnormal F1 | 0.7711 | 0.7711 |
| Train minus validation F1 | 0.0045 | 0.0045 |
| Validation sensitivity (abnormal recall) | 74.58% | 74.58% |
| Validation false negatives | 426 | 426 |

Validation prediction agreement was **3,115 of 3,115 recordings (100%)**, with 0 changed predictions.
