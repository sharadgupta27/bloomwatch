# Setup

Two supported paths. **Docker is the primary one** — it needs nothing on the host but
Docker itself, and it avoids the environment problems documented at the bottom of this
page. The conda path exists because GPU passthrough into containers on Windows requires
extra setup that not everyone will want.

---

## Path 1 — Docker (recommended)

### Requirements

- Docker Desktop (Windows/macOS) or Docker Engine + Compose v2 (Linux)
- For GPU training: the **NVIDIA Container Toolkit**. On Windows this means Docker
  Desktop with the **WSL2 backend** and a recent NVIDIA driver.

### Run

```bash
make build                  # build CPU + GPU images
make mlflow                 # tracking server -> http://localhost:5000
make ingest                 # build the Zarr cubes (resumable; see timing below)
make train                  # GPU training
make api                    # forecast API + map -> http://localhost:8000
make test                   # test suite inside the image
```

### Verify GPU passthrough before a long run

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

If that prints your GPU, `make train` will use it. If it errors, the toolkit is not
installed — `--gpus all` is then **silently ignored** and training falls back to CPU.
`docker/Dockerfile.gpu` prints its CUDA build at image-build time so this is visible
rather than discovered six hours later.

---

## Path 2 — conda (host)

```bash
conda env create -f environment.yml
conda activate hab
bash scripts/install_torch.sh     # CUDA wheel; see "PyTorch" below
pip install -e ".[all]"
pytest -m "not network" -q
```

### PyTorch is deliberately not in `environment.yml`

Putting this in a pip block **does not work**:

```yaml
- pip:
    - --extra-index-url https://download.pytorch.org/whl/cu124
    - torch>=2.4
```

`--extra-index-url` *adds* an index; pip still resolves the highest matching version
across all of them, and PyPI's CPU-only wheel usually wins. You end up with
`torch 2.x+cpu`, `torch.cuda.is_available()` returns `False`, and training silently runs
on CPU. This happened during development of this repo and cost a full environment rebuild.

`scripts/install_torch.sh` uses `--index-url`, which *replaces* the index list so the
PyTorch index is the only candidate source, then asserts `torch.cuda.is_available()` and
exits non-zero if it is not.

Pick `CUDA_TAG` to match your driver (`nvidia-smi`):

| Driver | Tag |
|---|---|
| ≥ 525 | `cu121`, `cu124` |
| ≥ 550 | `cu126`, `cu128` |
| ≥ 580 | `cu130` |

```bash
CUDA_TAG=cu130 PYTHON=$(which python) bash scripts/install_torch.sh
```

---

## Running the app

`make api` (Docker) or, on the host:

```bash
uvicorn serving.api.main:app --port 8000      # then open http://127.0.0.1:8000/app
```

The defaults serve the 21-lake registry from `data/processed/cubes`. For the full 46-lake
outlook with current imagery, point the API at the global registry and the live cubes and
weather (built by `artifacts/bulletin/refresh_global_live.sh <date>`, ~2–3 min per lake):

```bash
HAB_AOI=data/aoi/lakes_global.geojson \
HAB_CUBES=data/processed/cubes_global_live \
HAB_MET_ROOT=data/processed/met_live \
uvicorn serving.api.main:app --port 8000
```

What to expect:

- **Startup.** The server answers in ~10 s; a background thread then loads the model and
  builds the outlook (`HAB_WARMUP=0` turns this off). The first build for 46 lakes takes
  about a minute; after that the outlook is read from `data/cache/outlook.json` in 0.2 s
  and rebuilt only when a cube, a checkpoint or the validation summary changes.
- **Caches** live under `data/cache/` (`statics/` water masks, `outlook.json`, `erddap/`
  CoastWatch responses, `scenes/` downloaded scenes, capped at 2 GB). All are safe to
  delete; they rebuild.
- **Stored data first.** An area inside a monitored lake is answered from its archive when
  the archive holds the date asked for. With the live cubes weeks old, "newest imagery"
  still downloads; refresh them, or pick a date the archive holds.
- **Imagery age.** The outlook uses each lake's newest archived day. If the live cubes
  have not been refreshed, every lake is shown as out of date — that is the app being
  honest, not a fault.
- **Field-sample check and README figures** are offline steps:
  `python scripts/validate_insitu.py` (~7 min; `--from-matchups` to re-summarise) and
  `python scripts/make_readme_figures.py`.

All settings are listed in the README's *Configuration* section.

---

## Ingest timing and resumability

The full 5-lake, 2018–2024 ingest takes a few hours, dominated by per-scene HTTP
latency rather than bandwidth (measured: 12 MB/s aggregate available, but ~3.4 s per
scene cold).

It is **idempotent** — days already in a cube are skipped — so an interrupted run
resumes by re-running the identical command. To go lake by lake:

```bash
python pipelines/ingest.py --lakes erie_west --start 2018-01-01 --end 2024-12-31
```

Expect roughly 500 daily frames per lake across seven bloom seasons, at about
0.08 MB/day. The complete archive is well under 1 GB.

---

## Known environment pitfalls

These all cost real time during development. They are recorded so they cost you none.

**`conda create --clone` can destroy the target env.** A clone that fails on a missing
package-cache record (`SpecNotFoundInPackageCache`) removes the destination directory —
including one that a *later* command had legitimately created there. Build from
`environment.yml` instead of cloning.

**MLflow 3.x removed the filesystem tracking backend.** `file:./mlruns` now raises. The
config uses `sqlite:///mlruns/mlflow.db`; the parent directory must exist before the
store opens, which `pipelines/train.py` handles.

**Hydra ≥ 1.2 does not chdir by default.** `hydra.job.chdir` defaults to `False`, so
`Path.cwd()` inside a job is the *invocation* directory. Using it for output paths
scatters checkpoints into the repo root. Use
`HydraConfig.get().runtime.output_dir` instead.

**fsspec's HTTP filesystem has no default timeout.** A stalled connection to Azure Blob
blocks forever — observed as an ingest sitting for 2.5 hours with zero CPU and zero log
output. `habforecast.data.olci` sets explicit `sock_read`/`sock_connect`/`total`
timeouts and retries; `habforecast.data.ingest` adds a per-batch wall-clock budget and
recycles the process pool if a worker wedges below that layer.

**Windows uses `spawn`, not `fork`.** Process-pool workers cannot inherit state and the
entry point must be a real file — running a multiprocessing script through
`python - <<EOF` fails with `OSError: [Errno 22] Invalid argument: '<stdin>'`. Worker
state is rebuilt in `habforecast.data.parallel.init_worker`.

**Python 3.14 is too new for parts of this stack.** The environment pins 3.12.
Specifically: Hydra 1.3.x (every release through 1.3.7) passes a non-string `help` to
argparse, which 3.14 validates eagerly, so `pipelines/train.py` dies at argument parsing
with `ValueError: badly formed help string` before Hydra composes anything. 3.11 and
3.12 are what CI runs.

**MLflow upgrades change the SQLite schema.** A newer `mlflow` against an older
`mlruns/mlflow.db` refuses to start with *Detected out-of-date database schema*. Back
the file up and run `mlflow db upgrade sqlite:///$(pwd)/mlruns/mlflow.db`; the backup
from the last upgrade here is kept beside the database.

**AMP and the per-pixel window statistics.** `models.anomaly.WindowSummary` contracts the
time axis with a matmul, and under `16-mixed` that matmul ran in float16. Its backward
divides by the squared recency weight of a pixel observed only at the far end of the
window, which overflows float16; the gradient scaler then skips the step and halves its
scale until the scale underflows, and the parameters go NaN with the loss finite at every
step. The contraction now runs in float32 (`torch.autocast(enabled=False)`); if a future
change reintroduces a small-denominator division into an autocast region, expect the
same failure and look for it at the scaler, not the loss.

**On Windows, call the local API at `127.0.0.1`, not `localhost`.** `localhost` tries IPv6
first, and against a server bound to IPv4 every request then appears ~2 s slower than it
is — enough to send a performance investigation the wrong way.

**Bloom-season climatology cache.** `use_clim` models read a leave-one-year-out seasonal
mean per lead, cached per (day-of-year, year) as float16 full-grid frames under a
process-wide budget (`HAB_CLIM_CACHE_MB`, default 2048). The first epoch computes them
(~5 minutes for the three training lakes); later epochs read them back. Lower the budget
on a machine with less RAM; the run gets slower, not wrong.
