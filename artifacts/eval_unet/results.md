#### Per lake and lead time

| Lake | Split | Lead | RMSE (log₁₀) | vs persistence | vs climatology | CSI @20 | POD @20 | FAR @20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| victoria_ne | test | 1 d | 0.290 | -0.073 | -0.135 | 0.335 | 0.415 | 0.364 |
| victoria_ne | test | 3 d | 0.335 | -0.373 | -0.553 | 0.255 | 0.396 | 0.582 |
| victoria_ne | test | 5 d | 0.342 | -0.355 | -0.587 | 0.250 | 0.382 | 0.579 |
| victoria_ne | test | 7 d | 0.366 | -0.484 | -0.821 | 0.237 | 0.372 | 0.605 |
| balaton | val | 1 d | 0.340 | 0.158 | -0.056 | 0.056 | 0.083 | 0.851 |
| balaton | val | 3 d | 0.389 | 0.212 | -0.326 | 0.033 | 0.041 | 0.860 |
| balaton | val | 5 d | 0.385 | 0.250 | -0.361 | 0.027 | 0.037 | 0.913 |
| balaton | val | 7 d | 0.413 | 0.281 | -0.465 | 0.019 | 0.024 | 0.905 |

#### Model vs baselines (mean over lakes)

| Lead | Model RMSE | Persistence RMSE | Climatology RMSE |
|---:|---:|---:|---:|
| 1 d | 0.315 | 0.325 | 0.301 |
| 3 d | 0.362 | 0.362 | 0.303 |
| 5 d | 0.363 | 0.369 | 0.301 |
| 7 d | 0.389 | 0.394 | 0.306 |

_Skill is the Murphy skill score: 1 = perfect, 0 = no better than the baseline, < 0 = worse. Persistence is strongest at short lead; climatology does not degrade with horizon._