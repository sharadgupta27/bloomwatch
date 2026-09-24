#### Per lake and lead time

| Lake | Split | Lead | RMSE (log₁₀) | vs persistence | vs climatology | CSI @20 | POD @20 | FAR @20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| balaton | val | 1 d | 0.309 | 0.303 | 0.125 | 0.020 | 0.020 | 0.436 |
| balaton | val | 3 d | 0.350 | 0.363 | -0.072 | 0.017 | 0.017 | 0.537 |
| balaton | val | 5 d | 0.349 | 0.385 | -0.116 | 0.007 | 0.007 | 0.803 |
| balaton | val | 7 d | 0.368 | 0.429 | -0.164 | 0.006 | 0.006 | 0.570 |
| victoria_ne | test | 1 d | 0.241 | 0.255 | 0.213 | 0.293 | 0.316 | 0.198 |
| victoria_ne | test | 3 d | 0.242 | 0.283 | 0.189 | 0.262 | 0.287 | 0.249 |
| victoria_ne | test | 5 d | 0.246 | 0.298 | 0.178 | 0.259 | 0.288 | 0.283 |
| victoria_ne | test | 7 d | 0.248 | 0.318 | 0.163 | 0.258 | 0.289 | 0.295 |

#### Model vs baselines (mean over lakes)

| Lead | Model RMSE | Persistence RMSE | Climatology RMSE |
|---:|---:|---:|---:|
| 1 d | 0.275 | 0.325 | 0.301 |
| 3 d | 0.296 | 0.362 | 0.303 |
| 5 d | 0.297 | 0.369 | 0.301 |
| 7 d | 0.308 | 0.394 | 0.306 |

_Skill is the Murphy skill score: 1 = perfect, 0 = no better than the baseline, < 0 = worse. Persistence is strongest at short lead; climatology does not degrade with horizon._

#### Advisory-threshold ranking at 20 mg/m³

_ROC-AUC and PR-AUC for "chlorophyll exceeds the threshold at the target date". PR-AUC is the one to read: blooms are rare, and a constant forecast scores its prevalence. A model that cannot rank better than persistence has not learned anything persistence did not already know._

| Lake | Split | Lead | Bloom fraction | ROC-AUC | PR-AUC | persistence ROC / PR | climatology ROC / PR |
|---|---|---:|---:|---:|---:|---:|---:|
| balaton | val | 1 d | 0.014 | 0.830 | 0.149 | 0.774 / 0.099 | 0.869 / 0.079 |
| balaton | val | 3 d | 0.021 | 0.793 | 0.110 | 0.719 / 0.082 | 0.793 / 0.076 |
| balaton | val | 5 d | 0.020 | 0.752 | 0.073 | 0.755 / 0.069 | 0.773 / 0.066 |
| balaton | val | 7 d | 0.021 | 0.783 | 0.116 | 0.703 / 0.082 | 0.810 / 0.084 |
| victoria_ne | test | 1 d | 0.078 | 0.934 | 0.616 | 0.933 / 0.611 | 0.901 / 0.409 |
| victoria_ne | test | 3 d | 0.082 | 0.929 | 0.576 | 0.929 / 0.571 | 0.900 / 0.403 |
| victoria_ne | test | 5 d | 0.079 | 0.925 | 0.544 | 0.924 / 0.545 | 0.901 / 0.397 |
| victoria_ne | test | 7 d | 0.080 | 0.926 | 0.541 | 0.927 / 0.545 | 0.900 / 0.393 |

#### Leave-lake-out vs. temporal split (test, mean over lakes)

_New lake: whole lakes held out. New year: every lake (balaton, erie_west, okeechobee, taihu, victoria_ne) trained on seasons up to the cut-off and scored on the years after it. The gap between the columns is what a lake outside the registry costs._

| Lead | New lake: vs persistence | New lake: vs climatology | New year: vs persistence | New year: vs climatology |
|---:|---:|---:|---:|---:|
| 1 d | 0.255 | 0.213 | 0.301 | 0.190 |
| 3 d | 0.283 | 0.189 | 0.444 | 0.314 |
| 5 d | 0.298 | 0.178 | 0.345 | 0.311 |
| 7 d | 0.318 | 0.163 | 0.470 | 0.369 |

#### Temporal split, per lake and lead time

| Lake | Split | Lead | RMSE (log₁₀) | vs persistence | vs climatology | CSI @20 | POD @20 | FAR @20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| balaton | val | 1 d | 0.271 | 0.455 | 0.090 | 0.049 | 0.049 | 0.000 |
| balaton | val | 3 d | 0.365 | 0.450 | -0.139 | 0.056 | 0.056 | 0.000 |
| balaton | val | 5 d | 0.331 | 0.418 | -0.243 | 0.000 | 0.000 | — |
| balaton | val | 7 d | 0.349 | 0.382 | -0.168 | 0.031 | 0.031 | 0.000 |
| erie_west | val | 1 d | 0.179 | 0.379 | 0.143 | 0.353 | 0.550 | 0.503 |
| erie_west | val | 3 d | 0.209 | 0.382 | 0.079 | 0.380 | 0.487 | 0.366 |
| erie_west | val | 5 d | 0.209 | 0.364 | 0.097 | 0.397 | 0.538 | 0.398 |
| erie_west | val | 7 d | 0.204 | 0.452 | 0.119 | 0.372 | 0.494 | 0.399 |
| okeechobee | val | 1 d | 0.132 | 0.478 | 0.921 | 0.985 | 0.995 | 0.011 |
| okeechobee | val | 3 d | 0.135 | 0.589 | 0.907 | 0.986 | 0.999 | 0.013 |
| okeechobee | val | 5 d | 0.123 | 0.508 | 0.930 | 0.985 | 0.998 | 0.013 |
| okeechobee | val | 7 d | 0.168 | 0.390 | 0.860 | 0.965 | 0.992 | 0.027 |
| taihu | val | 1 d | 0.338 | 0.222 | 0.175 | 0.035 | 0.050 | 0.895 |
| taihu | val | 3 d | 0.242 | 0.193 | 0.235 | 0.119 | 0.155 | 0.662 |
| taihu | val | 5 d | 0.501 | 0.056 | -0.267 | 0.016 | 0.045 | 0.976 |
| taihu | val | 7 d | 0.310 | 0.319 | -0.180 | 0.005 | 0.019 | 0.994 |
| victoria_ne | val | 1 d | 0.202 | 0.232 | 0.482 | 0.535 | 0.643 | 0.239 |
| victoria_ne | val | 3 d | 0.208 | 0.310 | 0.482 | 0.492 | 0.608 | 0.279 |
| victoria_ne | val | 5 d | 0.206 | 0.335 | 0.456 | 0.501 | 0.638 | 0.300 |
| victoria_ne | val | 7 d | 0.206 | 0.309 | 0.485 | 0.488 | 0.624 | 0.310 |
| balaton | test | 1 d | 0.228 | 0.425 | 0.195 | 0.000 | 0.000 | — |
| balaton | test | 3 d | 0.291 | 0.344 | 0.235 | 0.200 | 0.200 | 0.000 |
| balaton | test | 5 d | 0.214 | -0.077 | 0.713 | 0.500 | 0.500 | 0.000 |
| balaton | test | 7 d | 0.275 | 0.469 | 0.408 | 0.000 | 0.000 | — |
| erie_west | test | 1 d | 0.184 | 0.094 | 0.357 | 0.024 | 0.024 | 0.270 |
| erie_west | test | 3 d | 0.173 | 0.244 | 0.037 | 0.096 | 0.115 | 0.639 |
| erie_west | test | 5 d | 0.171 | 0.322 | -0.006 | 0.138 | 0.152 | 0.409 |
| erie_west | test | 7 d | 0.186 | 0.294 | 0.123 | 0.151 | 0.217 | 0.669 |
| okeechobee | test | 1 d | 0.125 | 0.859 | 0.908 | 0.968 | 0.993 | 0.026 |
| okeechobee | test | 3 d | 0.128 | 0.818 | 0.893 | 0.956 | 0.997 | 0.041 |
| okeechobee | test | 5 d | 0.165 | 0.791 | 0.827 | 0.955 | 0.995 | 0.041 |
| okeechobee | test | 7 d | 0.141 | 0.710 | 0.892 | 0.972 | 0.990 | 0.018 |
| taihu | test | 1 d | 0.283 | 0.014 | 0.034 | 0.198 | 0.303 | 0.636 |
| taihu | test | 3 d | 0.155 | 0.408 | 0.338 | 0.189 | 0.249 | 0.562 |
| taihu | test | 5 d | 0.252 | 0.370 | 0.049 | 0.106 | 0.180 | 0.795 |
| taihu | test | 7 d | 0.168 | 0.496 | 0.242 | 0.041 | 0.048 | 0.780 |
| victoria_ne | test | 1 d | 0.296 | 0.113 | -0.547 | 0.297 | 0.342 | 0.304 |
| victoria_ne | test | 3 d | 0.229 | 0.407 | 0.067 | 0.315 | 0.408 | 0.421 |
| victoria_ne | test | 5 d | 0.227 | 0.316 | -0.029 | 0.302 | 0.489 | 0.560 |
| victoria_ne | test | 7 d | 0.215 | 0.383 | 0.179 | 0.295 | 0.384 | 0.439 |