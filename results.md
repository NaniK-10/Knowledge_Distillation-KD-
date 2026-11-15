# Knowledge Distillation Evaluation (Test N=3096, Seeds=[0, 1, 2])

## Predictive Accuracy

| Model                     |             MAE |             MSE |            RMSE |              R2 |
| ------------------------- | --------------: | --------------: | --------------: | --------------: |
| Teacher ensemble          |          0.3140 |          0.2202 |          0.4693 |          0.8334 |
| Arm 1: Stand-alone (NLL)  | 0.3332 ± 0.0018 | 0.2433 ± 0.0018 | 0.4932 ± 0.0018 | 0.8160 ± 0.0014 |
| Arm 2: Naive KD (MSE mix) | 0.3335 ± 0.0016 | 0.2361 ± 0.0015 | 0.4859 ± 0.0016 | 0.8215 ± 0.0012 |
| Arm 3: UAR-BD             | 0.3264 ± 0.0004 | 0.2336 ± 0.0007 | 0.4833 ± 0.0007 | 0.8234 ± 0.0005 |

## Computational Efficiency

| Model                        |                 Parameters | Median (ms) | Mean (ms) | p95 (ms) |
| ---------------------------- | -------------------------: | ----------: | --------: | -------: |
| Teacher ensemble (5 experts) |                    222,410 |       0.958 |     1.026 |    1.402 |
| Arm 1: Stand-alone (NLL)     | 2,722 deploy / 4,834 train |       0.055 |     0.065 |    0.091 |
| Arm 2: Naive KD (MSE mix)    | 2,722 deploy / 4,834 train |       0.054 |     0.058 |    0.079 |
| Arm 3: UAR-BD                | 2,722 deploy / 4,834 train |       0.055 |     0.058 |    0.068 |

Compression Ratio (Teacher / Student): 81.7x deploy (2,722 params) | 46.0x train (4,834 params)
