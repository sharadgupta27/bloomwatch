# Public chlorophyll data for bloom forecasting

A survey of what can actually be obtained, measured against the live services on
**2026-08-28**, not quoted from documentation. Every latency and record length below was
checked with a request; where a portal refused, the refusal is recorded.

The question behind it: the model was trained on three lakes because the ingest path was
expensive, and three lakes was the measured ceiling on accuracy. Anything that makes more
lakes and more years cheap is worth more than any change to the network.

---

## Usable without an account

| Resolution | Product | Record | Access | Verified |
|---|---|---|---|---|
| **300 m** | Sentinel-3 OLCI L2 WFR | 2018-04 → **2026-02-25** | Planetary Computer STAC, granule download + swath resample | newest granule confirmed by STAC search |
| **300 m** | OLCI chlorophyll L3, sectored (S3A + S3B) | **rolling 90 days** | NOAA CoastWatch ERDDAP, bbox subset | 443 sector datasets; newest 2026-08-27 |
| **750 m** | VIIRS chlorophyll, science quality, sectored | **2012-01 → 2025-08** | NOAA CoastWatch ERDDAP, bbox subset | Lake Erie 2024-08-14: 45.5% valid, median 11.6 mg/m³ |
| **2 km** | VIIRS + OLCI gap-filled (DINEOF) | 2018-01 → 2026-08 | NOAA CoastWatch ERDDAP | Erie and Taihu: 14/14 days, 74–85% coverage |
| **4 km** | OLCI global daily (S3A + S3B) | 2019-06 → 2026-08 | NOAA CoastWatch ERDDAP | ✓ |
| **4 km** | VIIRS science quality, global daily | 2012-01 → 2026-08 | NOAA CoastWatch ERDDAP | ✓ |
| **4 km** | ESA Ocean Colour CCI, merged multi-sensor | **1997 → 2022** | CEDA HTTP / OPeNDAP, no login | directory listing and file names confirmed |

525 chlorophyll datasets are published on the CoastWatch ERDDAP alone. The three
properties that make them useful here, and that a granule archive does not have:

- **Server-side subsetting.** One lake for one year is a bounding-box request returning a
  few megabytes. The same period from L2 granules is hundreds of gigabytes of swath, of
  which a lake-sized crop is about a thousandth.
- **Already gridded and quality-masked**, so ingest does no swath resampling.
- **No credentials**, so a container works out of the box.

## Requires a free account

| Product | Why it matters | Barrier |
|---|---|---|
| **Copernicus Data Space** — Sentinel-3 OLCI L2, all levels | The same granules as Planetary Computer but **hours** behind instead of six months. Catalogue query on 2026-08-28 returned a product acquired at **18:08 UTC that day**. | Catalogue is public; downloads need OAuth2 (`client_id=cdse-public`, username/password). Unauthenticated download returns 401. |
| **NASA CyAN** — cyanobacteria index, 300 m, ~2000 US lakes | Purpose-built for this exact problem: a cyanobacteria index rather than a general chlorophyll retrieval. | Earthdata Login. A direct `.tif` request returns HTTP 200 carrying a 10 KB **HTML login page**, which is worth knowing before wiring it into a pipeline. |
| **NASA OB.DAAC** L2/L3 for every ocean-colour mission | Deepest and most authoritative archive. | Earthdata Login. |
| **Copernicus Marine (CMEMS)** | Operational 300 m / 1 km / 4 km ocean colour. | Account; the public dataset API endpoint tried here returned 404. |

## Not usable

- **Sentinel-2 MSI (10–60 m)** and **Landsat (30 m)** resolve small water bodies, but
  neither publishes a chlorophyll product. Deriving one is a separate research problem
  with its own validation burden, not an ingest change.

---

## What was built on this

`habforecast.data.coastwatch` addresses the ERDDAP ladder through one product registry
(`PRODUCTS`), and `pipelines/ingest_erddap.py` writes any rung into the existing Zarr cube
format. `python pipelines/ingest_erddap.py --list-products` prints the table with the
trade-offs attached.

Two uses:

1. **Live forecasting** (`olci-300m`) — native resolution, one day behind, the first
   choice for a recent area drawn in the app. See `inference.adhoc`. Availability is
   intermittent: on 2026-09-23 no `noaacwS3?OLCIchlaSector*` dataset was listed, on
   2026-11-26 it served imagery from 2026-11-23. When it does not answer, the app reads
   the granule path instead — the EUMETSAT Data Store (hours behind, same product and
   quality control as training) when `eumdac` credentials are set, else Planetary Computer.
2. **Training** (`viirs-750m`, `viirs-4km`, `dineof-2km`) — the deep records. The 300 m
   OLCI rung cannot train anything: its window is 90 days.

### Choosing a rung for training

Grid size decides this more than anything else. The forecaster downsamples three times, so
an input smaller than about 16 px has nothing left at the coarse head, and it fails
outright below 8 px.

| Rung | Grid for a 20 km lake | Grid for the smallest lake in the registry | Trainable |
|---|---|---|---|
| 300 m | 133 px | 82 px | yes |
| 750 m | 53 px | 34 px | yes |
| 2 km | 20 px | 13 px | yes, degraded |
| 4 km | 10 px | 7 px | yes, but no skill (see below) |

Below ~16 px the third downsample has nothing left to halve. `avg_pool2d` raises on a 1×1
input rather than returning an empty tensor, so this originally crashed outright at 7×7 —
the model could not even be *evaluated* on the coarse rung. `_pool_if_possible` now stops
halving at 1×1, which is inert above 16 px and therefore changes nothing for the grids the
existing checkpoints were trained on.

Running is not the same as working: the 4 km arm trains and evaluates fine and has no
useful skill. See the measured comparison below.

### What changes between rungs, besides the pixel

A comparison across rungs is only controlled if it holds the sensor fixed. `viirs-750m`
and `viirs-4km` are the same instrument, the same retrieval and the same period, which
makes that pair a genuine resolution experiment. Comparing either to the 300 m OLCI cubes
is **not** controlled: different instrument, different retrieval, different quality
masking. Skill against persistence and climatology is still comparable, because each
configuration is scored against its own baselines on lakes it has never seen — but
absolute error is not.

---

## More than chlorophyll: auxiliary input layers

The forecaster was univariate from the start — chlorophyll, its observation mask, a water
mask and the day of year. Everything else that drives a bloom had to be inferred from
where chlorophyll had been.

**The cheap place to find more is the granule already being downloaded.** A Sentinel-3
OLCI L2 WFR product carries 34 assets and the ingest opened four of them. Measured
2026-08-30, three more are worth their read cost, all at 300 m, over inland water, with
no additional credential:

| layer | from | what it adds that chlorophyll cannot |
|---|---|---|
| **mci** | `oa10/oa11/oa12-reflectance` | height of the 709 nm peak above a 681–754 nm baseline. CHL_NN is a neural-net inversion that degrades on dense surface scum; MCI is three-band arithmetic with no inversion in it, so a high MCI beside an unremarkable CHL_NN is the signature of a scum the retrieval under-read. |
| **tsm** | `tsm-nn` | suspended sediment — reddens water like a bloom does, and is the commonest false positive in a river plume. |
| **kd490** | `trsp` | light attenuation: the depth of the lit layer a bloom can grow in, and an independent read on turbidity. |

Three more were added 2026-09-25, aimed at a question the first three cannot answer —
*what kind* of bloom this is. MCI and TSM separate green water from brown water; nothing
so far separates a cyanobacterial bloom from a harmless algal one, which is the
distinction that decides whether a bloom is a public-health problem.

| layer | from | what it adds |
|---|---|---|
| **ci** | `oa08/oa10/oa11-reflectance` | the Wynne–Stumpf Cyanobacteria Index (NOAA's operational Lake Erie index), reported sign-flipped so larger means more bloom. Eukaryotic algae fluoresce at 681 nm and cyanobacteria do not, so the curvature there reverses between them. MCI keys on near-infrared brightness and calls any dense surface algae a bloom; CI is what says whether it is the dangerous kind. |
| **pc620** | `oa06/oa07/oa08-reflectance` | depth of the 620 nm absorption trough below a 560–665 nm baseline. Phycocyanin is the one pigment no eukaryotic alga carries, so this is direct taxonomic evidence rather than an inference from bloom shape. **Exploratory** — the feature is shallow, OLCI's 620 nm band is 10 nm wide, and at 300 m over a small lake the dip may sit inside the noise. |
| **adg443** | `iop-nn` | absorption by coloured dissolved organic matter and detritus. TSM catches mineral sediment; this catches brown peat-stained water, where nothing is suspended and TSM reads low. A different false positive from the one TSM covers. |

**Read cost is not the sum of the per-layer costs.** The three spectral indices name nine
asset slots but only six distinct bands (MCI and CI share Oa10/Oa11; CI and pc620 share
Oa08), and the reader caches each band per scene. Use `aux_layers.asset_reads(keys)` to
cost an ingest — summing `ASSET_COST` overstates the full spectral set by a third.

Measured on Lake Erie's western basin, 2023 bloom season: each layer covers **42.6% of
water-days**, identical to chlorophyll's own coverage — they share the swath and the QC
mask, so they are not a sparser signal, and their ranges are physically sensible
(MCI −0.006…+0.054; TSM 0.13–300 g m⁻³; Kd490 0.08–10 m⁻¹).

Enable with `pipelines/ingest.py --aux mci,ci,pc620,tsm,kd490,adg443` and a matching
`data.aux=[...]`.
Cubes without them train exactly as before — a missing layer arrives as an all-zero
channel with an all-zero mask, which the model already reads as "never observed".

### Does it help? Not measured yet

The plumbing is built and tested; **whether these layers improve skill is an open
question, and this document will not answer it until a run does.** The comparison is
`scripts/aux_experiment.py`, which runs five arms: chlorophyll alone (`chl-only`), plus
MCI (`chl-mci`), plus the original three (`chl-aux3`), plus the three spectral indices
(`chl-taxo`), and plus everything (`chl-all`). All are trained on **the same cubes, the
same lakes, the same period and the same architecture**, differing only in how many
channels reach the trunk. That is a cleaner comparison than the resolution experiment
could be, because no sensor, retrieval or masking changes between arms.

**Each arm runs three seeds.** This project has measured roughly 0.07 of seed-to-seed
variance in skill, which is larger than any effect three extra channels are likely to
have, so a single-seed difference between arms is not a result. The script reports mean
± half-range and an arm wins only if its mean clears the others by more than the spread.

The result that would count is skill against persistence and climatology on the *held-out*
lakes. A gain on the validation lake alone is what sparse extra channels being used to
memorise the training lakes looks like, and is the outcome to expect if this does not
work.

**The blocker is cubes, not code.** Only `cubes_aux` carries auxiliary layers, and it has
two lakes — not enough for a held-out-lake result. The 46-lake `cubes_global` ingest does
not pass `--aux`. A real verdict needs a re-ingest of at least the 21-lake
`lakes.geojson` registry with the full `--aux` set.

### What was tried and does not work

| candidate | why not | measured |
|---|---|---|
| **Water temperature** (NOAA ACSPO, geo-polar blended, Coral Reef Watch) | these are *sea*-surface products and inland water is masked out of them | 630 cells returned over western Lake Erie, **0 valid**, in all three |
| **VIIRS Kd490 / KdPAR** | no inland coverage at the lake | HTTP 404 for the bounding box |
| **PACE OCI** | resolution, not access: even Level-2 is ~1.2 km | L3 published at 0.1° and 4 km; L2 swath 1.2 km, so a 12 km lake is 10 px across |

The PACE result deserves stating plainly, because it is the source most people would
expect to help. It is a real, current mission and its hyperspectral reflectance is
genuinely unique. But this project has already measured what a 4 km rung buys on lakes, in
a controlled comparison against the same sensor at 750 m: **essentially no skill**.

**Access is no longer the blocker.** As of 2026-09-25 this machine authenticates to
Earthdata via `~/.netrc`, verified end to end: `PACE_OCI_L2_BGC_NRT` for the previous day
returned HTTP 200 and 9.9 MB of netCDF. `habforecast.data.pace` now accepts either
`EARTHDATA_TOKEN` or a `machine urs.earthdata.nasa.gov` netrc entry, and knows the
Level-2 collections (`l2-aop`, `l2-iop`, `l2-bgc` and their NRT variants).

**Resolution still is.** At 1.2 km a 12 km lake is 10 pixels across, below the ~16 the
forecaster's third downsample needs, so PACE remains unusable as a forecast *input
channel* for most of the registry. The role that survives the arithmetic is **reference
data**: the new `ci` and `pc620` OLCI indices are 300 m proxies for phytoplankton
community composition, and PACE L2 IOP/AOP actually measures it. On the handful of large
lakes PACE resolves, it can say whether those proxies track what they claim to — which is
the one validation those indices otherwise have no way to get. That is a separate exercise
from the forecaster, and it has not been run.

`python -m habforecast.data.pace --product l2-aop` reports all of this from the live
catalogue.

---

## Choosing the product and the model for the water selected

Every request used to take one path: OLCI at 300 m, forecast by the checkpoint trained on
three inland lakes. That is right for Lake Erie and wrong twice over 300 km into the Gulf
of Mexico, where the 300 m sectors do not reach and no model here has ever been evaluated.
`habforecast.water` now classifies the drawn area first and `habforecast.routing` turns
that into an explicit plan, which the app shows before it fetches anything (`/api/route`).

### Classifying the water

Offline, deterministic, ~20 ms, from one packed Natural Earth ocean polygon
(`data/reference/ne10_ocean.npz`, 1.3 MB, built by `scripts/build_coast_reference.py`).
A point-in-polygon test answers inland-or-marine; the distance to the polygon's boundary —
with the artificial map edges at ±180° and ±90° dropped — answers how far from shore.
Verified against Natural Earth's separate coastline layer at ten sites: agreement within
0.5 km, which is why the second 10 MB file is not shipped.

Two axes, because they decide different things and do not always agree:

| | decides | values |
|---|---|---|
| **Setting** (hydrology) | which products exist for the area | `inland`, `coastal`, `open-ocean` |
| **Optical class** (physics) | which retrieval is trustworthy | `case-2`, `case-1` |

The Curonian Lagoon is why they are separate: marine by hydrology, optically complex by
physics. Two other cases the test set pins down — the Caspian and Aral seas, which Natural
Earth maps *as* ocean and which are endorheic, so they are named explicitly and treated as
very large lakes; and mid-Pacific near the antimeridian, where the polygon's closing seam
would read as a coastline if it were not filtered out.

### The policy, and the measurements behind it

| Water | Product | Model | Why |
|---|---|---|---|
| A monitored lake, opened as one (`lake_id`) | its own ingested cube — nothing fetched | `olci-300m-clim` (`best.ckpt`) | it has the multi-year archive the climatology checkpoint needs (`routing.registry_route`); ~1 s |
| Inland, and coastal within 40 km of shore (Case-2) | OLCI 300 m near-real-time, archive fallback | `olci-300m` | CHL_NN is the retrieval built for optically complex water; band-ratio products mask it away |
| Open ocean (Case-1) | VIIRS 4 km global, OLCI 4 km fallback | **none** — labelled persistence baseline | the 300 m and 750 m sectors do not reach here, and the 4 km checkpoint measured no skill |

Three things this encodes that the single path could not:

1. **A checkpoint never leaves its product.** The grid resolution is now the route's, not
   always 300 m — an open-ocean request builds a 4 km grid, which is honest about what was
   measured and shrinks the payload instead of blowing thirteen real numbers up into a
   blocky picture.
2. **A stale product never silently serves a live request.** Route order is a preference,
   not an instruction: a step is skipped when a later one is fresher by more than one input
   window. This was found by running it — asking the 750 m route for today returned imagery
   from the previous August, because CoastWatch stopped publishing that product on
   2025-08-19 while the 300 m sectors were a day behind.
3. **Where nothing is validated, nothing is claimed.** Open ocean returns imagery plus
   persistence, labelled as persistence. A baseline whose error is characterised beats a
   model whose is not.

### What routing cannot fix

It cannot invent a coastal model. **Every training body in this project is a lake**, so
every marine route carries that disclosure on the result. Routing picks the best available
combination; the missing evidence is a training job, not a dispatch decision — and the
obvious one is to train the 750 m recipe (15 bodies, 19.6 k body-days) on OLCI 300 m data,
which would put the better-trained model and the better retrieval in the same place for
the first time.

---

## Serving it reliably

Measured on 2026-08-29, when NOAA's ERDDAP proxy spent an afternoon answering 502 after
sixty seconds. One ad-hoc forecast took **375 seconds** and returned imagery from January.

The wall clock is almost entirely network, so the failure modes are the performance story:

| | before | after |
|---|---|---|
| forecast during the outage | 375 s | **20 s** |
| repeat of the same area | full price | **1.3 s** |
| second forecast during an outage | full price again | archive only, no live attempt |

Four changes, in order of how much each was worth:

1. **Tell "no data" apart from "no answer".** ERDDAP returns 404 for an empty selection
   and 502 when the proxy gives up. The anchor ladder treated both as "this fortnight was
   unusable" and stepped back a fortnight — asking a dead server six more questions at a
   minute each. `SourceUnavailableError` now separates them, and a transport failure
   switches to the archive instead of walking.
2. **A circuit breaker.** Discovering an outage costs a full budget; discovering it once
   per lake, while a user clicks down a list, costs the afternoon. Two failed sector reads
   cut the live source off for three minutes. `/health` reports the state.
3. **A finished-forecast cache**, keyed by area, anchor and source. The validation sidebar
   makes repeats normal, and a repeat is now 1.3 s.
4. **Concurrent sector reads and a response cache.** S3A and S3B are independent requests
   to one server; they were serial. Griddap responses are cached for 12 hours, since
   imagery for a past fortnight does not change.

A retry loop was the obvious fix and, on its own, the wrong one: three attempts at a
60-second failure is a 186-second failure, measured. Retries are bounded by a total
budget (`GRID_BUDGET_S`, 30 s for a live request), and bulk ingest opts out because its
windows are legitimately minutes long.

**Monitored lakes skip the network entirely (2026-11-26).** Opening a lake from the app's
outlook sends its `lake_id`, and the forecast is built from the lake's ingested cube
(`inference.adhoc.forecast_from_archive`): ~1 s instead of 17 s or more. Per-lake water
masks are kept on disk (`HAB_STATIC_CACHE`) and the outlook itself (`HAB_OUTLOOK_CACHE`),
so a restart serves the outlook in 0.2 s; both are keyed by the cubes' signatures and
rebuild themselves after a data refresh. The README's *Serving speed* table has the
before-and-after numbers.

**What is not fixed:** the archive fallback is ~25 s and stays serial, because
`data/parallel.py` records the measurement that threads do not help — h5py and gzip hold
the GIL, and 24 threads gave an effective concurrency of 1.2. A process pool is what works
there, and paying spawn cost per web request to speed up the fallback is the wrong trade.

---

## Retraining: what it actually bought

Run 2026-08-28/29. All three arms use the same architecture (`AnomalyForecaster`, 333 K
parameters) and the same leave-lake-out protocol; only the data changes. Skill is the
Murphy score against each configuration's own persistence and climatology baselines,
averaged over the held-out lakes.

| Arm | Training lakes | Lake-days | test d1 | d3 | d5 | d7 |
|---|---|---|---|---|---|---|
| OLCI 300 m *(original)* | 3 | ~2 k | +0.046 | +0.037 | +0.042 | +0.062 |
| **VIIRS 750 m** | **15** | **19.6 k** | **+0.075** | **+0.065** | **+0.084** | **+0.074** |
| VIIRS 4 km | 15 | 17.7 k | −0.015 | +0.002 | −0.032 | −0.011 |

*(skill vs persistence; the 300 m row is a single held-out lake, the others four)*

And against climatology, which the 300 m model never beat at any lead:

| Arm | test d1 | d3 | d5 | d7 |
|---|---|---|---|---|
| OLCI 300 m | −0.008 | −0.089 | −0.122 | −0.151 |
| **VIIRS 750 m** | **+0.191** | **+0.103** | **+0.095** | **+0.012** |
| VIIRS 4 km | +0.006 | −0.179 | −0.082 | −0.179 |

On the one lake held out of **both** runs, Lake Victoria's Winam Gulf:

| lead | 300 m / 3 lakes | 750 m / 15 lakes |
|---|---|---|
| 1 d | +0.046 | **+0.137** |
| 3 d | +0.037 | **+0.167** |
| 5 d | +0.042 | **+0.145** |
| 7 d | +0.062 | **+0.154** |

Three findings, in order of how much they matter:

1. **The data was the ceiling, as the earlier evaluation suggested.** Five times the
   lakes and ten times the lake-days roughly tripled skill against persistence on the same
   held-out lake, with the same 333 K-parameter network. No architectural change in this
   project has come close to that.
2. **Climatology is beaten now.** Every earlier model lost to a seasonal climatology at
   every lead, which is the single most damning result a forecast system can have. The
   750 m model beats it at all four leads on the test lakes.
3. **Resolution matters, and 4 km is not enough.** The 750 m and 4 km arms are the same
   sensor, the same lakes and the same period, so the comparison is controlled: 750 m wins
   on every held-out lake and every lead, and 4 km has essentially no skill against
   persistence at all. At 4 km a lake is 7–31 pixels, and the coarse arm's better
   *validation* score alongside its collapse on test is what overfitting to a handful of
   pixels looks like.

### Where it still fails

**Hartbeespoort Dam is negative at every lead in both arms** (−0.14 at 750 m, −0.24 at
4 km). It is 12 km across, so 34 px at 750 m and 7 px at 4 km, and VIIRS returns a valid
retrieval on 0.4% of its pixels. The registry rule "big enough to resolve" was written
before the ingest and this lake violates it in practice; it is kept in the test split
because a held-out set that only contains lakes the model handles is not a test.

**VIIRS masks turbid inland water heavily.** Mean valid-pixel coverage runs from 28% on
the Curonian Lagoon down to 2.6% on Taihu — one of the most bloom-affected lakes in the
world, and nearly invisible to this sensor. That is an argument for the OLCI 300 m rung,
whose CHL_NN retrieval is designed for exactly this water, and against reading the VIIRS
result as the last word.

**These numbers are not comparable in absolute terms to the OLCI runs.** Different
instrument, different retrieval, different masking. What is comparable is skill against
baselines computed within each configuration — which is what the tables above report.
