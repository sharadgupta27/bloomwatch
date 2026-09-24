# Bloom Watch

**Short-range forecasting of cyanobacterial harmful algal blooms in inland waters, from Sentinel-3 OLCI.**

The Python package is `habforecast` (`src/habforecast/`); Bloom Watch is the project, the
API and the map app built on it.

Most satellite bloom tools *map* what already happened. This one forecasts 1–7 days
ahead, on a held-out lake it has never seen, and reports its skill against the baselines
that actually matter. A water manager cannot act on a map of yesterday; they can act on
"the western basin will cross the advisory threshold on Thursday."

> **Advisory only.** Satellite chlorophyll-a is not a direct cyanotoxin measurement.
> See [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) for intended use and limitations.

---

## Why this matters

Cyanobacterial blooms shut off drinking water for 400,000 people in Toledo, Ohio in 2014.
They close beaches, kill fish, poison livestock, and cost the US alone billions annually.
Lake Taihu supplies water to roughly 10 million people and blooms every summer. Lake
Victoria supports around 40 million.

Monitoring is mostly reactive: a bloom is sampled after someone sees it. Sentinel-3 OLCI
observes every lake on Earth larger than a few km², daily, at 300 m, for free — and it
carries bands (620/665/709 nm) designed for exactly this. The gap is not data. It is a
forecasting system with an honest error bar.

---

## What this repository is

A complete, production-shaped ML system that runs end to end on a laptop:

```
Sentinel-3 OLCI (STAC)  ──▶  windowed read + QC  ──▶  daily Zarr datacube
                                                            │
                        ┌───────────────────────────────────┤
                        ▼                                   ▼
              masked sequence dataset              drift monitoring
                        │                          (obs / dist / geographic)
                        ▼
     anomaly forecaster (+ seasonal climatology input)  ·  ConvLSTM  ·  U-Net
                   persistence  ·  climatology baselines
                        │
                        ▼
   leave-lake-out + temporal evaluation ──▶ MLflow ──▶ FastAPI + map + advisory feed
```

| Layer | Choice | Why |
|---|---|---|
| Data source | Planetary Computer STAC, anonymous | No credentials needed to reproduce |
| Storage | Zarr, one cube per lake | Chunked, appendable, ~0.08 MB per lake-day |
| Config | Hydra (training only) | Sweeps; ingest stays plain CLI |
| Tracking | MLflow | Runs, params, split composition, checkpoints |
| Data versioning | DVC | Cubes and pipeline DAG |
| Orchestration | Docker Compose | `make ingest` / `make train` / `make api` |
| Serving | FastAPI + inline map | Forecast JSON + PNG overlays; advisory feed as GeoJSON and RSS |
| Scheduled job | `pipelines/advisory.py` (compose service, cron-able) | Daily advisories to disk, same code as the live endpoint |
| Monitoring | Custom drift module | Observation / distribution / **geographic** drift |
| CI | GitHub Actions | Lint, tests, config contracts, image build + API smoke test |

---

## Three findings worth reading

These are the parts that were not obvious, and each is pinned by a regression test.

### 1. The obvious QC mask deletes 100% of your data

OLCI's `WQSF` flag field behaves counter-intuitively over lakes. On a real Lake Erie
granule:

| Flag | Frequency | |
|---|---|---|
| `WATER` | **0.00%** | never set over inland water — it is an open-ocean flag |
| `LAND` | 75.85% | **set over inland water too** |
| `INLAND_WATER` | 54.88% | |
| `INLAND_WATER & LAND` | 40.65% | exactly the set with a finite `CHL_NN` retrieval |

The textbook mask — require `WATER`, reject `LAND` — returns an empty dataset. The first
ingest here did precisely that and produced zero valid pixels across every scene.
Correct approach: **require `INLAND_WATER`, do not reject `LAND`**.

### 2. Two more flags are bloom-correlated, and rejecting them biases the training set

`TURBID_ATM` (22.8% of pixels) and `OC4ME_FAIL` (7.0%) are commonly rejected. Both
correlate with dense blooms — surface scum raises reflectance enough to trip the
turbid-atmosphere test, and `OC4ME_FAIL` just means the open-ocean band-ratio algorithm
failed, which it always does in case-2 water. That is *why* we read `CHL_NN` instead.
Rejecting them preferentially deletes the strongest blooms. Both are deliberately kept;
[`STRICT_REJECT_FLAGS`](src/habforecast/data/qc.py) exists to quantify the difference.

### 3. The workload looks I/O-bound and is not

Scene reads are three HTTP range reads against Azure Blob, so a thread pool is the
obvious choice. Measured effective concurrency with 24 threads: **~1.2** — h5py's HDF5
calls and gzip decompression hold the GIL. Switching to processes cut cold-cache
per-scene wall time from **5.8 s to 3.4 s**.

Two other ingest optimisations, both measured rather than assumed:

- **Tie-point windowing.** A full granule is 4090 × 4865. Locating the AOI via the ~1 MB
  column-subsampled tie-point grid, then range-reading only the intersecting HDF5 chunks,
  reduces a scene to **0.1% of the granule**. Strided reads (`arr[::16, ::16]`) defeat
  this completely — they touch every chunk and take 52 s.
- **QC before geolocation.** `geo-coordinates` is 61 MB against 5 MB for `chl-nn`.
  Reordering the reads so QC runs first lets ~50% of scenes be rejected before the
  expensive read happens at all.

---

## The evaluation is the point

**Split by lake, never randomly.**

| Split | Lakes | Rationale |
|---|---|---|
| train | Erie (western basin), Okeechobee, Taihu | Temperate + subtropical, all hypereutrophic |
| val | Balaton | Narrow elongated basin — different morphology |
| **test** | **Victoria (Winam Gulf)** | **Tropical, no seasonal cycle, different continent** |

A random split would put 2019-07-14 Lake Erie in train and 2019-07-15 in validation.
Those frames share most of their pixels; the score would measure memorisation. Adjacent
300 m pixels are strongly autocorrelated, so random *spatial* tiling fails the same way.
Holding out whole lakes answers the only question an operational user has: *will this
work on my lake?* Lake Victoria is deliberately the hardest available case.

**Always report skill against both baselines.** Persistence ("tomorrow looks like today")
is very strong at 1-day lead and degrades with horizon. Climatology (per-pixel seasonal
mean) is weak at 1 day and *stops* degrading. Where those curves cross is where a learned
model can earn its keep. Reporting RMSE alone, or against only one baseline, flatters the
model.

**Report event scores, not just regression.** An operator acts on a threshold crossing,
not a chlorophyll value. `POD`, `FAR`, `CSI` and `F1` are computed at 10 / 20 / 50 mg/m³.

**Everything is masked.** With 60–75% of target pixels unobserved, an unmasked metric is
dominated by pixels no satellite ever saw.

### Results

Served model: `anomaly` forecaster with the lake's own seasonal climatology as an input
(`conf/model/anomaly_clim.yaml`, 0.34M params, 28 epochs early-stopped, 1,475 training
samples from Erie + Okeechobee + Taihu), scored with flip test-time augmentation.
Regenerate with `make evaluate results`; the full tables, including ROC-AUC / PR-AUC at
every advisory threshold, are in [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).

| Lake | Split | Lead | RMSE (log₁₀) | vs persistence | vs climatology | POD @20 | FAR @20 | CSI @20 | PR-AUC @20 (model / persistence) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Balaton | val | 1 d | 0.309 | **+0.303** | +0.125 | 0.020 | 0.436 | 0.020 | 0.149 / 0.099 |
| Balaton | val | 3 d | 0.350 | **+0.363** | −0.072 | 0.017 | 0.537 | 0.017 | 0.110 / 0.082 |
| Balaton | val | 5 d | 0.349 | **+0.385** | −0.116 | 0.007 | 0.803 | 0.007 | 0.073 / 0.069 |
| Balaton | val | 7 d | 0.368 | **+0.429** | −0.164 | 0.006 | 0.570 | 0.006 | 0.116 / 0.082 |
| Victoria | **test** | 1 d | 0.241 | **+0.255** | **+0.213** | 0.316 | 0.198 | 0.293 | 0.616 / 0.611 |
| Victoria | **test** | 3 d | 0.242 | **+0.283** | **+0.189** | 0.287 | 0.249 | 0.262 | 0.576 / 0.571 |
| Victoria | **test** | 5 d | 0.246 | **+0.298** | **+0.178** | 0.288 | 0.283 | 0.259 | 0.544 / 0.545 |
| Victoria | **test** | 7 d | 0.248 | **+0.318** | **+0.163** | 0.289 | 0.295 | 0.258 | 0.541 / 0.545 |

Two split strategies, because they answer different questions (mean over test lakes):

| Lead | New lake: vs persistence | New lake: vs climatology | New year: vs persistence | New year: vs climatology |
|---:|---:|---:|---:|---:|
| 1 d | +0.255 | +0.213 | +0.301 | +0.190 |
| 3 d | +0.283 | +0.189 | +0.444 | +0.314 |
| 5 d | +0.298 | +0.178 | +0.345 | +0.311 |
| 7 d | +0.318 | +0.163 | +0.470 | +0.369 |

*New lake* holds out whole lakes (Victoria). *New year* keeps every lake and trains on
2018–2022, validates on 2023, tests on 2024–2025 (`data.split_mode=temporal`); its
climatology and normalisation are built from the training years only. The gap between the
columns is what a lake outside the registry costs.

#### How it got here

The previous served model (the same architecture without the climatology input) scored
+0.046 → +0.062 against persistence on Victoria and *lost* to a per-pixel seasonal
climatology at every lead (−0.008 → −0.151). The diagnosis in the earlier version of this
section was that a model which sees a fortnight cannot know where the season is heading.
Feeding it the leave-one-year-out seasonal mean for each target date, and letting it learn
per pixel and per lead how far to move its persistence anchor toward that mean, is the
whole change (`src/habforecast/models/anomaly.py`, change 5). Skill against persistence
on the held-out lake went from +0.05 to +0.26–0.32, and it now beats climatology at every
lead — which no OLCI checkpoint here did before, and which the 15-lake VIIRS 750 m
checkpoint (+0.137 → +0.154) does not match either.

#### What this actually says

1. **The model generalises to the held-out tropical lake.** Skill against persistence
   grows with lead (+0.26 → +0.32) as it should, and climatology — flat at 0.27 RMSE
   across leads — is beaten by 0.16–0.21 everywhere. A random train/test split would have
   reported a similar-looking number for the wrong reason; this one is earned on a lake,
   a continent and a climate the model never saw.

2. **The ranking metrics are the honest caveat.** PR-AUC for "exceeds 20 mg/m³" is only
   level with persistence on Victoria (0.616 vs 0.611 at 1 day, 0.541 vs 0.545 at 7). The
   model reduces error everywhere but does not rank the bloomiest pixels much better than
   yesterday's image does — the smooth, bounded delta that makes it generalise is also what
   stops it from sharpening the extremes. Event scores, not RMSE, are where the next gain
   has to come from.

3. **Balaton still loses to climatology at 3–7 days** (−0.07 → −0.16) while beating
   persistence by +0.36–0.43. Balaton is mesotrophic with a strong, stable seasonal cycle:
   its own climatology is a very good forecast, and the model is only allowed to lean part
   of the way toward it. POD at 20 mg/m³ is near zero there because the lake almost never
   crosses it (bloom fraction 1.4–2.1%).

4. **Without the climatology the model is persistence.** Measured with the input withheld
   (`pipelines/evaluate.py --no-clim`): −0.003 → +0.033 on Victoria. An ad-hoc area drawn
   on the map has no archive to build a climatology from, so `habforecast.routing` serves
   it the previous checkpoint (`models/olci-300m-nohistory.ckpt`, +0.046 → +0.062) and says
   why. The registry lakes get the new one.

#### What would fix the rest

Stated as hypotheses, not claims — none of these have been run:

- **A sharper event head.** Score the threshold crossing directly (a classification
  head at each advisory level, trained with focal loss) alongside the regression, so the
  ranking metrics are optimised rather than inherited.
- **More and more diverse training lakes.** Three lakes cannot span the world's limnology;
  the VIIRS experiment showed what fifteen buys.
- **Weather forcing.** ERA5 wind and temperature drive bloom advection and growth, and the
  model currently sees none of it. Needs a Copernicus CDS account, so it is not in the
  no-credentials reproduction path.

---

## The central difficulty: most of the data is missing

Measured across the ingested archive (2018–2025, bloom season only):

| Lake | Daily frames | Grid | Water px | Bloom-season days with data | Per-pixel density | Median chl-a | P90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Erie (W. basin) | 540 | 267×298 | 42,895 | 44.3% | 51.9% | 4.1 | 10.6 |
| Okeechobee | 735 | 212×180 | 14,941 | 43.2% | 47.0% | 25.2 | 50.2 |
| Taihu | 492 | 266×252 | 25,524 | 29.0% | 36.8% | 14.1 | 22.6 |
| Balaton | 399 | 166×260 | 5,931 | 41.5% | 54.8% | 6.4 | 13.6 |

*Bloom-season days with data* = fraction of calendar days in the lake's bloom window with
at least one QC-passing scene. *Per-pixel density* = fraction of (day, water-pixel) slots
actually observed. Chlorophyll in mg/m³.

So **roughly 3 in 5 observations do not exist**, and the gaps are not random — they
cluster around the storm systems that also drive bloom dynamics. Okeechobee's median of
25 mg/m³ against Erie's 4.1 also shows why per-lake evaluation matters: these are not
interchangeable distributions.

This shapes every design decision downstream:

- Frames are placed on a **regular daily calendar**, so a 3-day gap and a 30-day gap are
  distinguishable.
- Every sample carries **explicit mask channels**; the model is told what it can trust
  rather than having to infer it from a sentinel value.
- The **loss is masked and normalised by the mask**, not by tensor size. Dividing by
  `numel()` would make the loss depend mostly on how cloudy a sample was, and the
  optimiser would learn that shrinking predictions toward zero is the cheapest way down.
- Every model **predicts a delta from the last observed value**, which makes persistence
  the default behaviour and forces the network to earn any departure from it. The served
  model also blends that anchor toward the lake's seasonal climatology, by a learned
  weight that is zero wherever no climatology exists.

---

## Quickstart

```bash
git clone https://github.com/<your-account>/bloom-watch.git
cd bloom-watch
```

### Docker (recommended)

```bash
make build
make mlflow                 # tracking UI on :5000
make ingest                 # build Zarr cubes  (~2-3 h, resumable)
make train                  # GPU training; needs the NVIDIA Container Toolkit
make api                    # forecast API + map on :8000
```

GPU training in a container requires the NVIDIA Container Toolkit — on Windows that
means Docker Desktop with the WSL2 backend. Without it, `--gpus all` is silently ignored;
`docker/Dockerfile.gpu` therefore prints its CUDA build at image-build time so the
failure is visible rather than a six-hour CPU run.

### Conda fallback

```bash
conda env create -f environment.yml
conda activate hab
bash scripts/install_torch.sh          # CUDA wheel — see below
pip install -e ".[all]"

python pipelines/ingest.py --start 2018-01-01 --end 2024-12-31 --workers 12
python pipelines/train.py                                   # anomaly_clim, leave-lake-out
python pipelines/train.py data.split_mode=temporal          # the second split
python pipelines/evaluate.py --checkpoint models/best.ckpt --out artifacts/eval --tta
python pipelines/evaluate.py --checkpoint models/best-temporal.ckpt --out artifacts/eval_temporal --split-mode temporal --tta
python scripts/make_results_table.py --scores artifacts/eval/scores.json --temporal artifacts/eval_temporal/scores.json --model-card docs/MODEL_CARD.md
python pipelines/advisory.py                                # today's GeoJSON + RSS feed
uvicorn serving.api.main:app --reload                       # /app, /forecast, /advisory, /advisory/feed.xml
```

Python 3.11 or 3.12. Hydra 1.3 does not parse arguments on 3.14 (see `docs/SETUP.md`).

**PyTorch is deliberately not in `environment.yml`.** Putting
`--extra-index-url .../cu124` in a pip block does not work: pip still resolves the
highest matching version, which is the CPU-only wheel on PyPI, and you silently train on
CPU. `scripts/install_torch.sh` uses `--index-url` so the PyTorch index is the only
source, then asserts `torch.cuda.is_available()`.

---

## Running on 4 GB of VRAM

Developed and trained on an RTX 3050 Laptop (4 GB, sm_86). What makes it fit:

- Lake grids are small by construction — largest is 370 × 351 at 300 m.
- 128 × 128 random crops for training; full grids only at evaluation, batch size 1.
- AMP (`16-mixed`) plus gradient accumulation: effective batch 16 from a physical batch 8.
- Models under 3M parameters. ConvLSTM at 48 hidden channels, U-Net at base 32.
- Observations cached in RAM as float16 — log10 chlorophyll spans about −1 to 2.5, far
  coarser than float16 precision, and it halves the footprint.

Total archive for 5 lakes × 7 years is well under 1 GB.

---

## Repository layout

```
src/habforecast/
  aoi.py                  deterministic per-lake analysis grid
  data/olci.py            windowed OLCI reader (tie-point AOI lookup, HTTP timeouts)
  data/qc.py              WQSF masking - the inland-water traps
  data/aux_layers.py      extra input layers from the same granule (MCI, TSM, Kd490)
  data/pace.py            PACE OCI: gated reader, and why it is not the default
  data/resample.py        swath -> grid nearest-neighbour with distance cutoff
  data/store.py           Zarr cube: idempotent append, merge-on-backfill
  data/parallel.py        process-pool worker protocol (Windows spawn-safe)
  data/ingest.py          monthly batching, pool recycling on a wedged worker
  features/dataset.py     masked sequence sampling on a regular daily calendar
  features/datamodule.py  leave-lake-out and temporal splits, train-only normalisation
  models/anomaly.py       the served forecaster: instance-normalised, persistence-anchored,
                          smooth bounded delta, seasonal-climatology input and anchor blend
  models/convlstm.py      ConvLSTM (gradient-checkpointed) + U-Net forecasters
  models/baselines.py     persistence, leave-one-year-out climatology
  models/lit.py           masked Huber, per-lead skill logging, flip TTA, model_inputs()
  eval/metrics.py         regression + event scores, ROC/PR-AUC, Murphy skill -- all streaming
  advisory.py             forecasts -> advisory levels -> GeoJSON + RSS
  water.py                inland / coastal / open-ocean and Case-1 vs Case-2, offline
  routing.py              which product and which checkpoint, and why - as data
  inference/predictor.py  archive or live-STAC inference; persistence baseline
  insitu.py               public in-situ measurements near a point
pipelines/
  ingest.py               bulk ingest CLI (argparse; idempotent, resumable)
  train.py                Hydra + Lightning + MLflow
  evaluate.py             per lake x lead scoring against both baselines
  monitor.py              drift report against the training baseline
  advisory.py             the scheduled forecast: every lake -> GeoJSON + RSS on disk
serving/                  FastAPI app + map UI (chart.js: per-pixel time series,
                          field-data sidebar, GIS map-sheet export)
monitoring/drift.py       observation / distribution / geographic drift
scripts/                  env setup, CUDA torch install, results-table renderer,
                          in-situ survey and index builder, auxiliary-layer survey,
                          the chlorophyll-only vs multi-layer experiment, and the
                          Natural Earth coastline reference builder
data/reference/           packed ocean polygon the water classifier reads (1.3 MB)
docker/                   CPU (serve, ingest) and GPU (train) images
conf/                     Hydra config tree
tests/                    500+ unit tests + live-network integration tests
```

---

## Inland or coastal: the tool picks the data and the model, and shows its working

Draw an area and the app classifies it before it fetches anything — inland lake, coastal
bay, open ocean — and pairs it with a data product and a checkpoint that were measured
together. The decision, and the argument for it, appear in the panel before you spend
thirty seconds on a forecast.

| Water selected | Imagery | Model |
|---|---|---|
| Inland lake, or coastal within 40 km of shore | Sentinel-3 OLCI 300 m (CHL_NN) | 300 m checkpoint; the 750 m one offered as an alternative where the area is ≥ 16 px at 750 m |
| Open ocean beyond the shelf | VIIRS / OLCI global 4 km | **none** — a labelled persistence baseline |

Two reasons the obvious answer is not the answer. The 750 m checkpoint is measurably the
better forecaster — roughly triple the skill against persistence, and the only one here
that beats a seasonal climatology — but its band-ratio retrieval masks turbid inland water
heavily (2.6% valid pixels on Taihu), and this tool exists for that water. And the 4 km
checkpoint scored *negative* skill against persistence, so over open ocean the app returns
imagery and a persistence baseline rather than dressing a measured non-result as a
forecast. Every marine route also states that no checkpoint here was trained on marine
water. See [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md#choosing-the-product-and-the-model-for-the-water-selected).

---

## Limitations

Stated plainly, because a model card that only lists strengths is not a model card.

- **Chlorophyll-a is a proxy.** It does not distinguish cyanobacteria from harmless algae,
  and it is not toxin concentration. Phycocyanin-based indices would be better; in-situ
  validation data for them is far too sparse to train on.
- **Ground truth is satellite-derived.** The model forecasts the *satellite-observed
  field* — the same thing NOAA's operational Lake Erie HAB bulletin does. In-situ data is
  a check on that field, never a training target, and there is not enough of it to be
  more: 6,185 laboratory chlorophyll values worldwide in GLORIA, of which 422 fall inside
  a modelled lake and inside the ingested period. Clicking a forecast pixel plots
  whatever has been measured within 10 km of it beside the prediction
  (`habforecast.insitu`, `docs/VALIDATION_DATA.md`), which is honest about how often the
  answer is "nothing in this window".
- **Optical retrieval fails in specific conditions**: thick cloud, sun glint, high solar
  zenith, and very shallow water where bottom reflectance contaminates the signal.
- **Small effective dataset.** 29–44% of bloom-season days carry any usable scene, which
  constrains model capacity and is why a U-Net control is evaluated alongside the
  ConvLSTM rather than assumed inferior.
- **The geographic-novelty check is under-powered.** Measured against the three training
  lakes, tropical Lake Victoria scores 0.89 and temperate Lake Balaton 0.86 — the metric
  cannot currently distinguish them, because the spread across three already-diverse
  reference lakes absorbs the difference. The `seasonal_amplitude` descriptor alone
  separates them (0.080 vs 0.126); the pooled RMS z-distance does not. Reported here
  rather than quietly dropped, because a monitoring metric that looks reassuring while
  being blind is worse than no metric.
- **Planetary Computer's Sentinel-3 archive starts in 2018**, so 2016–2017 are unavailable
  through this path.
- **The mirror lags real time by roughly 6–7 months.** Measured 2026-08-28: the newest
  OLCI L2 WFR granule on Planetary Computer was 2026-02-25. This is a property of the
  mirror, not of Sentinel-3 — the mission is operational near-real time, and EUMETSAT
  publishes NRT products within ~3 hours of acquisition. A nightly integration test
  asserts the lag has not silently grown worse.
- **The app therefore reads a second source for live forecasts.**
  `habforecast.data.coastwatch` fetches NOAA CoastWatch daily Level-3 OLCI chlorophyll
  over ERDDAP: 300 m, S3A and S3B, no credentials, server-side bounding-box subsetting,
  and one day behind (measured 2026-08-28). `inference.adhoc` picks between the two by
  the date requested. It is not a replacement for the archive — it keeps only a rolling
  90-day window, so training and evaluation still run on the Planetary Computer path —
  and it differs in kind, not only in age: the composites arrive pre-masked, so this
  project's own WQSF quality control and its flag-frequency drift telemetry do not apply,
  and the product metadata does not name the retrieval algorithm. Every ad-hoc result
  reports which source served it, and says so in its caveats.
  Copernicus Data Space carries the same granules within hours and would be the better
  long-term ingest, but its downloads require a registered account; its catalogue is
  public and was used to confirm the mission latency above.

---

## License

Apache-2.0. Sentinel-3 data © Copernicus / ESA, free and open. Not affiliated with ESA,
EUMETSAT, NOAA, or Microsoft.
