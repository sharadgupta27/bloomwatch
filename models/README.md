# Served checkpoints

`best.ckpt` is the default (`HAB_CHECKPOINT`) and what the registry-lake endpoints
(`/forecast`, `/advisory`, `/advisory/feed.xml`, `pipelines/advisory.py`) use. The ad-hoc
`/app` path does not load one checkpoint: `habforecast.routing` classifies the water body
and loads whichever of these was trained on data like it, resolving the file against
`HAB_MODEL_ROOT` (`/models` in the container).

| file | architecture | trained on | held-out skill vs persistence (d1 -> d7) |
|---|---|---|---|
| `best.ckpt` | anomaly + seasonal climatology input (`artifacts/anomaly_clim`, epoch 16) | OLCI 300 m, 3 lakes, leave-lake-out | **+0.255, +0.283, +0.298, +0.318** (Victoria, TTA); beats climatology at every lead |
| `best-temporal.ckpt` | same (`artifacts/anomaly_clim_temporal`, epoch 24) | OLCI 300 m, 5 lakes, years <= 2022 | +0.301, +0.444, +0.345, +0.470 (2024-25, mean over lakes) -- the second split number, not served |
| `olci-300m-nohistory.ckpt` | anomaly (`artifacts/anomaly`, epoch 20) | OLCI 300 m, 3 lakes | +0.046, +0.037, +0.042, +0.062 -- what an area with no archive gets |
| `viirs-750m.ckpt` | anomaly (`artifacts/multires`) | VIIRS 750 m, 15 lakes | +0.075, +0.065, +0.084, +0.074 |
| `ens_met/` (3 checkpoints + `calibration.json`) | anomaly + climatology + ERA5 forcing + exceedance head, seeds 1337/1, 2 | OLCI 300 m, 3 lakes, leave-lake-out | +0.219, +0.276, +0.246, +0.284 (ensemble); **PR-AUC@20 0.662 → 0.597 vs persistence 0.611 → 0.545** -- read advisory probabilities from this |
| `ens_clim/` | `best.ckpt` + two more seeds of it | same | +0.234, +0.221, +0.252, +0.255 -- what the served configuration scores on average |
| `onnx/best/` | ONNX export of `best.ckpt` (`scripts/export_onnx.py`) | -- | identical to `best.ckpt` within 4e-5 log10; served by the `serve-onnx` image |
| `convlstm-best-18.ckpt` | convlstm (`artifacts/convlstm`, epoch 18) | OLCI 300 m, 3 lakes | -0.422, -0.771, -1.225, -0.678 |

`best.ckpt` reads the lake's own multi-year archive to build a leave-one-year-out seasonal
climatology for each target date, and blends its persistence anchor toward it. That is
where its skill comes from: with the input withheld it measures -0.003 to +0.033 against
persistence, below `olci-300m-nohistory.ckpt`. An area drawn on the map has no archive, so
the router serves it the no-history checkpoint and says why. Pointing `HAB_CHECKPOINT` at
`olci-300m-nohistory.ckpt` reproduces the previous deployment exactly.

`viirs-750m.ckpt` was the strongest checkpoint here before the climatology input and is
still offered as an alternative on clear water large enough for 750 m pixels. Its
retrieval is a band-ratio algorithm that masks optically complex water heavily -- 2.6%
valid pixels on Taihu -- and CoastWatch stopped publishing the 750 m sector product on
2025-08-19, so it answers historical questions, not today's. The app shows both facts.

The ConvLSTM served here until 2026-08-28. It scores better on the *validation* lake than
the no-history model and much worse on the held-out one, and the `/app` endpoint forecasts
arbitrary user-chosen locations -- every one of which is, by definition, a lake the model
has never seen. Kept for the record.

A directory of checkpoints is an **ensemble**: point `HAB_CHECKPOINT` (or `--checkpoint`)
at it and every member runs, the forecast is their mean and the exceedance probability
their (recalibrated) mean. A directory holding `manifest.json` is an ONNX export and runs
without torch. `best.scores.json` is the held-out evaluation `scripts/promotion_gate.py`
gates the next candidate against; it is replaced only by a promotion.

None of these files is tracked by git (see .gitignore); they are build outputs of
`pipelines/train.py` and are reproduced by `make train`.
