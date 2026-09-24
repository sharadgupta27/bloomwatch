# Model card — Bloom Watch (`habforecast`)

Following the model-card framework of Mitchell et al. (2019).

---

## Model details

| | |
|---|---|
| **Name** | habforecast |
| **Version** | 0.1.0 |
| **Type** | Spatiotemporal forecasting; masked dense regression over image sequences |
| **Architectures** | Served: `anomaly` forecaster (~0.34M params; instance-normalised window summary, persistence anchor blended toward the seasonal climatology, smooth bounded delta). Also trained and reported: ConvLSTM encoder–forecaster (~0.36M) and U-Net over stacked time channels (~0.85M). |
| **Input** | 14 daily frames of log₁₀ chlorophyll-a at 300 m + per-frame observation masks + static water mask + day-of-year encoding + the lake's leave-one-year-out seasonal climatology at each target date (with its own availability mask). **Optionally** auxiliary physical layers from the same granule (MCI, suspended matter, Kd490) — see `data/aux_layers.py`; the served checkpoint uses none of them. |
| **Output** | Predicted log₁₀ chlorophyll-a at 1, 3, 5 and 7 day lead times |
| **Training data** | Sentinel-3 OLCI Level-2 WFR (`CHL_NN`), 2018–2024, bloom season only. A second served checkpoint is trained on VIIRS 750 m over 15 lakes; see *Served checkpoints* below. |
| **License** | Apache-2.0 |

Every architecture predicts a **delta from the last observed value**, so an untrained
network is exactly a persistence forecaster. This is pinned by a test
(`test_predicts_persistence_at_init`) because it is the inductive bias the whole design
depends on. The served model additionally moves that anchor toward the seasonal
climatology by a learned, spatially varying weight that is exactly zero where no
climatology exists — so on a lake with no history it is the persistence-anchored model
it extends (`tests/unit/test_climatology_input.py`).

---

## Intended use

**Intended.** Research and decision *support* for lake managers, public-health agencies,
and water utilities. Prioritising where to send limited in-situ sampling capacity.
Retrospective analysis of bloom dynamics. A worked reference for building masked
spatiotemporal forecasting on sparse satellite observations.

**Not intended.** Issuing public health advisories without in-situ confirmation.
Regulatory compliance monitoring. Any use where a false negative endangers people.
Drinking-water treatment decisions as a sole input. Lakes smaller than roughly
3 × 3 km, where 300 m pixels are dominated by shoreline mixing.

**Out of scope.** Marine/coastal blooms (different optical regime and different
algorithm), toxin concentration prediction, species identification, and any lake outside
the registry without re-running the geographic-novelty check.

**The app will let you draw a coastal or open-ocean area, and it will tell you this.**
`habforecast.routing` classifies the selected water body before it fetches anything and
attaches the disclosure to the result: no checkpoint here was trained or evaluated on
marine water, so a forecast there is an extrapolation outside its measured domain. Where
the only product that covers the area is a 4 km grid, no learned model runs at all — the
4 km checkpoint measured negative skill against persistence, so the app returns the
imagery and a labelled persistence baseline instead. Refusing to draw the area would hide
a limitation; stating it on the result does not.

---

## Served checkpoints

The deployment serves more than one model, because the right model depends on the water.
`habforecast.routing` pairs each with the product it was trained on and cannot move it off
that product: the network is convolutional, so its receptive field is a fixed number of
*pixels*, and running the 750 m checkpoint on 300 m imagery would make every feature it
learned 2.5× the wrong size.

| Checkpoint | Grid | Training bodies | Skill vs persistence (test, d1/d3/d5/d7) | Beats climatology | Served |
|---|---|---|---|---|---|
| `models/best.ckpt` (climatology input) | OLCI 300 m | 3 lakes | **+0.255 / +0.283 / +0.298 / +0.318** | **yes** | yes — registry lakes (`/forecast`, `/advisory`), which have the archive it reads |
| `models/best-temporal.ckpt` | OLCI 300 m | 5 lakes, years ≤ 2022 | +0.301 / +0.444 / +0.345 / +0.470 (held-out 2024–25, mean over lakes) | yes | no — the temporal-split number, kept for the comparison |
| `models/olci-300m-nohistory.ckpt` | OLCI 300 m | 3 lakes | +0.046 / +0.037 / +0.042 / +0.062 | no | yes — ad-hoc areas drawn on the map, which have no archive |
| `models/viirs-750m.ckpt` | VIIRS 750 m | 15 lakes | +0.075 / +0.065 / +0.084 / +0.074 | **yes** | yes — offered where ≥16 px at 750 m |
| `artifacts/multires/anomaly-viirs4km.ckpt` | VIIRS 4 km | 15 lakes | −0.015 / +0.002 / −0.032 / −0.011 | no | **no** |

The climatology-aware checkpoint needs years of the lake's own archive to build its
input. With that input withheld it measures −0.003 → +0.033 against persistence on the
held-out lake, below the no-history checkpoint, so `habforecast.routing` keeps the
no-history one for ad-hoc areas and says so on the result.

The 750 m checkpoint was the better forecaster than the previous 300 m one and is *not*
the default. Two reasons, both measured. Its retrieval is a band-ratio algorithm that masks optically complex water
heavily — 2.6% valid pixels on Taihu, one of the most bloom-affected lakes in the world —
and this tool exists for exactly that water. And CoastWatch stopped publishing the 750 m
sector product on **2025-08-19**, so it can answer historical questions but not today's.
Both facts are shown next to the option in the app.

The 4 km checkpoint is catalogued and never served. It was trained and evaluated under the
same protocol as the others and has no skill; listing it without serving it is the point,
because that negative result is what justifies serving a baseline over open ocean instead.

---

## Factors

Performance varies systematically along these axes, and evaluation is stratified by them.

- **Lake** — results are reported per lake, never pooled. Pooling across lakes of
  different size and trophic state hides the failure mode this project cares about.
- **Lead time** — 1 to 7 days. Skill against persistence *increases* with lead time;
  skill against climatology decreases. Both are reported.
- **Observation density** — forecasts built on 2 observed input days are materially less
  reliable than those built on 12. The API returns the count so consumers can filter.
- **Season** — trained on bloom-season months only. Off-season behaviour is untested.
- **Trophic state** — training lakes are eutrophic to hypereutrophic. Oligotrophic lakes
  are out of distribution.

---

## Metrics

**Regression**, in log₁₀ space where error is approximately homoscedastic: RMSE, MAE,
bias, R². Reporting RMSE in raw mg/m³ would be dominated by a few surface-scum pixels.

**Event**, at 10 / 20 / 50 mg/m³ thresholds: POD, FAR, CSI, F1 at the threshold itself,
and ROC-AUC / PR-AUC for exceeding it, reported for the model and for both baselines. An
operator acts on a threshold crossing, not a concentration; PR-AUC is the one to read,
because blooms are rare and a constant forecast scores its prevalence.

**Skill**, Murphy skill score against **both** persistence and climatology. Either alone
flatters the model — persistence is near-unbeatable at 1 day, climatology is weak there
but does not degrade with horizon.

All metrics are **masked**: only pixels observed in the target frame and inside the water
mask contribute. With 60–75% of pixels unobserved, unmasked metrics are meaningless.

---

## Evaluation data

**Split by lake, never randomly** — the headline number.

| Split | Lakes | Why |
|---|---|---|
| train | Erie (western basin), Okeechobee, Taihu | Temperate + subtropical, hypereutrophic |
| val | Balaton | Different basin morphology (narrow, elongated) |
| test | Victoria (Winam Gulf) | Tropical, no seasonal cycle, different continent |

**And split by year** — the second number, for lakes already in the registry.

| Split | Years | Lakes |
|---|---|---|
| train | 2018–2022 | all five |
| val | 2023 | all five |
| test | 2024–2025 | all five |

Held-out years never see training years in their input windows, their climatology and
normalisation come from the training years only, and both are pinned by tests.

A random split would place consecutive days of the same lake on both sides of the
boundary. Those frames share most of their pixels, so the score would measure
memorisation. Adjacent 300 m pixels are strongly autocorrelated, so random spatial
tiling fails identically.

---

## Quantitative results

The section between the markers is written by `scripts/make_results_table.py` from the
evaluation output, never by hand, so the card cannot quote a number the pipeline did not
produce. Regenerate after every evaluation:

```bash
python pipelines/evaluate.py --checkpoint models/best.ckpt --out artifacts/eval --tta
python pipelines/evaluate.py --checkpoint models/best-temporal.ckpt --out artifacts/eval_temporal --split-mode temporal --tta
python scripts/make_results_table.py --scores artifacts/eval/scores.json --temporal artifacts/eval_temporal/scores.json --model-card docs/MODEL_CARD.md
```

<!-- results:begin -->
_Generated by `scripts/make_results_table.py` from `artifacts/eval/scores.json` on 2026-09-22. Do not edit by hand._

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
<!-- results:end -->

### Seeds, weather forcing and the exceedance head (2026-09-23)

Hand-written, unlike the block above, because none of these replaced the served
checkpoint. Leave-lake-out, held-out lake (Victoria), TTA, leads 1 / 3 / 5 / 7 days.
Sources: `artifacts/eval*/scores.json`.

| Model | Skill vs persistence | R² (log10) | PR-AUC@20 | Brier skill @20 |
|---|---|---|---|---|
| served `best.ckpt` (anomaly_clim, seed 1337) | 0.255 0.283 0.298 0.318 | 0.683 0.664 0.663 0.649 | 0.616 0.576 0.544 0.541 | — |
| anomaly_clim, seed 1 | 0.211 0.171 0.195 0.190 | 0.664 0.612 0.614 0.583 | 0.583 0.519 0.489 0.476 | — |
| anomaly_clim, seed 2 | 0.164 0.106 0.141 0.127 | 0.644 0.581 0.588 0.550 | 0.590 0.539 0.514 0.502 | — |
| **anomaly_clim, 3-seed ensemble** | 0.234 0.221 0.252 0.255 | 0.674 0.635 0.641 0.616 | 0.607 0.557 0.532 0.521 | — |
| + exceedance head, no weather (seed 1337) | 0.188 0.233 0.167 0.219 | 0.655 0.641 0.600 0.597 | 0.619 0.563 0.538 0.533 | 0.26 0.20 0.13 0.13 |
| **+ ERA5 + head, 3-seed ensemble, calibrated** | 0.219 0.276 0.246 0.284 | 0.667 0.661 0.638 0.631 | **0.662 0.624 0.609 0.597** | **0.35 0.31 0.29 0.29** |
| persistence | — | — | 0.611 0.571 0.545 0.545 | — |

What this says:

- **The served checkpoint is a fortunate seed.** Two more seeds of the identical
  configuration score 0.11–0.21 against persistence where it scores 0.26–0.32, and the
  three-seed ensemble sits between. Its headline numbers above are real but are the top
  of the configuration's range, not its expectation. Validation (Balaton) does not predict
  which seed wins on Victoria, so a seed cannot be chosen honestly after the fact.
- **The exceedance head alone does not rank better than persistence.** Weather forcing
  plus the head does, at every lead and in every one of the three seeds individually
  (+0.04 to +0.06 PR-AUC over persistence for the ensemble). Ranking the pixels that
  will cross the threshold needs information yesterday's image does not carry; ERA5 wind,
  temperature, light and rain supply some of it.
- **Weather does not improve the regression map.** Ensemble for ensemble it is level
  (better at 3 and 7 days, worse at 1 and 5), and against the served seed it fails the
  promotion gate on skill at 1, 5 and 7 days and on R² at 5 days. `best.ckpt` is therefore
  unchanged; `models/ens_met` is the model to read advisory probabilities from.
- **Calibration** is a two-parameter logistic map per lead and threshold fitted on the
  validation lake only (`habforecast/calibration.py`). Raw, the head was overconfident on
  Balaton (Brier skill −0.5 to −0.9) and S-shaped on Victoria; fitted on Balaton, it lifts
  Victoria's Brier skill from +0.23–0.30 to +0.29–0.35 and halves its calibration error
  (0.07 → 0.03). It does not remove the lake-to-lake base-rate difference: on Victoria,
  mid-range probabilities are now *under*-confident (p ≈ 0.35 verifies about 66% of the
  time), and Brier skill at 10 mg/m³, 7 days fell (0.46 → 0.40). The 50 mg/m³ level is
  served uncalibrated — the validation lake has too few exceedances to fit it.
- **Ensemble spread is not a usable uncertainty for the map.** Three seeds disagree by
  a quarter to a third of the ensemble's actual error (spread/RMSE 0.26–0.36). The
  calibrated exceedance probability is the uncertainty the advisory carries; the spread
  is reported but should not be read as an error bar.

---

## Ethical considerations

**False negatives are the dangerous error.** A missed bloom may mean people swim in or
drink contaminated water. The loss function is symmetric and therefore does *not* encode
this asymmetry; operational deployment should tune the decision threshold toward higher
POD at the cost of FAR, and the reported FAR should inform how much crying-wolf that buys.

**Chlorophyll-a is a proxy for the thing that matters.** It does not distinguish
cyanobacteria from harmless algae and it is not toxin concentration. A high chlorophyll
forecast is a reason to sample, not a reason to close a beach.

**Unequal data, unequal service.** In-situ validation data is abundant in North America
and Europe and sparse elsewhere — yet the lakes where blooms cause the most human harm
are often the least monitored. A model validated mainly on well-instrumented lakes may
serve under-monitored ones worse, precisely where it is needed most. Lake Victoria is
held out as the test lake partly to keep this visible.

**Automation bias.** A confident-looking map invites more trust than a 300 m satellite
retrieval deserves. Every API response carries the observation mask, the count of
observed input days, and a disclaimer for this reason.

---

## Caveats and recommendations

- **Ground truth is satellite-derived, not in-situ.** The Water Quality Portal returned
  124 chlorophyll samples for Lake Erie's western basin over nine years, and zero
  phycocyanin or microcystin. The model forecasts the *satellite-observed field* — the
  same target NOAA's operational Lake Erie HAB bulletin forecasts. In-situ data is a
  sanity check, not a training signal.
- **Retrieval fails in known conditions**: thick cloud, sun glint, high solar zenith,
  ice, and shallow water where bottom reflectance contaminates the signal.
- **QC choices are deliberate and consequential.** `TURBID_ATM` and `OC4ME_FAIL` are
  *retained* despite convention, because both correlate with dense blooms and rejecting
  them biases the training set toward clear water. `LAND` is retained because OLCI sets
  it over inland water. See `src/habforecast/data/qc.py`.
- **Small effective dataset.** ~26% temporal coverage in bloom season yields a few
  hundred usable frames per lake-year. This caps model capacity and is why a U-Net
  control is evaluated alongside the ConvLSTM rather than assumed inferior.
- **Archive starts in 2018** via Planetary Computer; 2016–2017 OLCI is not available
  through this path, and the mirror lags real time by ~6–7 months (measured 2026-08-28:
  newest granule 2026-02-25). **Training and evaluation use this path only.**
- **Live forecasts in the app may use a different source.** For recent dates
  `inference.adhoc` reads NOAA CoastWatch daily Level-3 OLCI chlorophyll, which is one
  day behind. This is a distribution the model was not trained on: the composites are
  pre-masked (so the WQSF screening in `data.qc` is bypassed) and the product metadata
  does not state the retrieval algorithm, where training used CHL_NN. Measured over Taihu
  on 2026-08-26 the source reported a median of 18.6 mg/m³, inside the training
  distribution for that lake, and the forecaster normalises each input window to its own
  mean and standard deviation, which cancels a systematic offset or scale difference but
  not a difference in spatial texture. **Skill has not been re-measured on this source** —
  the 90-day rolling window does not overlap the evaluation period. Treat NRT forecasts
  as operationally useful and quantitatively unvalidated. **As of 2026-09-23 that source is
  gone** (CoastWatch no longer lists any `noaacwS3?OLCIchlaSector*` dataset) **and has been
  replaced by the EUMETSAT Data Store** (`src/habforecast/data/eumetsat.py`): the same
  OLCI L2 WFR product the model trains on, with the same WQSF quality control, published
  hours after the overpass. It needs a free EUMETSAT account (`eumdac set-credentials`);
  without one every live path falls back to the Planetary Computer archive.
- **Skill on the live source, measured.** The five lakes' 2026 bloom seasons (March–
  22 September) were ingested from the Data Store (processing baseline 004; training used
  003) and scored with the temporal split: 2026 is in no checkpoint's training data, and
  its climatology input comes from earlier years only. `best.ckpt` beats persistence on
  every lake at every lead — mean skill +0.24 / +0.27 / +0.28 / +0.32 at 1 / 3 / 5 / 7
  days (`artifacts/eval_nrt2026_best`) — with R² 0.63–0.74 on Erie and Victoria. The weak
  spot is Victoria, where skill falls to +0.05–0.16 against +0.26–0.32 in the leave-lake-out
  archive evaluation while R² holds; `models/ens_met` does better there (+0.14–0.24) and
  ranks bloom pixels above persistence on most lakes and leads, measured *without* its
  weather input (2026 ERA5 had not arrived). A year effect and a baseline effect cannot
  be separated with one year, but nothing here suggests baseline 004 broke the model.
- **The served map's ranking is level with persistence.** PR-AUC for exceeding 20 mg/m³
  on the held-out lake is 0.616 against persistence's 0.611 at 1 day and 0.541 against
  0.545 at 7. The ERA5 + exceedance-head ensemble (`models/ens_met`) is the first model
  here that ranks better (0.662 / 0.597); see the section after the generated results.
- **Weather forcing lags real time.** ERA5 is published ~5 days behind, so a forecast
  issued today has the most recent days of its window masked. The model tolerates masked
  days by construction, but it was trained on complete windows and the cost of the gap is
  not measured. A forecast-weather feed would close it.
- **The 15-lake VIIRS 750 m checkpoint** (`models/viirs-750m.ckpt`, +0.137 → +0.154
  against persistence) was the strongest model here before the climatology input, and
  remains the alternative the app offers on clear water large enough for 750 m pixels.
  See `docs/DATA_SOURCES.md`.
- **Before applying to a new lake**, run `monitoring/drift.py`'s geographic-novelty check.
  The model will produce confident output for any lake handed to it, including ones
  nothing like its training set.

---

## References

- Mitchell et al. (2019), *Model Cards for Model Reporting*, FAT* '19.
- EUMETSAT, *Sentinel-3 OLCI Level-2 Water Product Notice* — WQSF flag semantics.
- NOAA GLERL, Lake Erie Harmful Algal Bloom Bulletin — the operational analogue.
- Murphy (1988), *Skill scores based on the mean square error*, Mon. Wea. Rev.
