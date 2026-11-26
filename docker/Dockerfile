# CPU image: ingest, evaluation, and the serving API.
#
# Measured: 2.01 GB. It was 3.48 GB before two fixes -- installing the CPU torch wheel
# before the project (see below), and installing only the `serve` extra so evidently's
# dependency chain (plotly 187 MB, pyarrow 156 MB, mlflow 62 MB, statsmodels 55 MB) stays
# out of a container whose job is to answer HTTP requests. torch's 750 MB is now the
# floor; going below it would mean exporting to ONNX and dropping torch from serving.
#
# Pure pip, no conda. Modern manylinux wheels cover the whole geospatial stack
# (rasterio, pyproj, h5py, zarr), so the conda layer that environment.yml needs on
# Windows buys nothing here and would triple the image size.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libexpat1/libgomp1 are runtime deps of the h5py and sklearn wheels respectively.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libexpat1 libgomp1 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the CPU torch build FIRST. pyproject lists `torch` as a dependency, so
# installing the project first resolves it from PyPI -- which serves the CUDA build and
# drags ~2.5 GB of nvidia-* wheels into an image that will never see a GPU. Taking it
# from the cpu index up front means the project install finds it already satisfied.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Dependency layer next so source edits do not invalidate the pip cache.
COPY pyproject.toml README.md ./
COPY src/habforecast/__init__.py src/habforecast/__init__.py
RUN pip install --no-cache-dir -e ".[serve]"

COPY src/ src/
COPY conf/ conf/
COPY pipelines/ pipelines/
COPY serving/ serving/
# The /dashboard route imports scripts/make_dashboard.py at request time to render the
# results page from whatever evaluation is currently on the artifacts mount. Without this
# the route 500s inside the container while working perfectly on the host.
COPY scripts/ scripts/
COPY data/aoi/ data/aoi/
# The packed ocean polygon the water classifier reads (1.3 MB). Baked in rather than
# mounted: without it every area classifies as "unknown" and the router falls back to its
# most conservative route, which is a working app that has quietly stopped doing the one
# thing this layer exists for.
COPY data/reference/ data/reference/
# Served at /model-card; the API description links to it.
COPY docs/MODEL_CARD.md docs/MODEL_CARD.md

# Run as non-root: the container writes only to mounted volumes.
RUN useradd --create-home --uid 10001 hab && chown -R hab:hab /app
USER hab

ENV PYTHONPATH=/app/src \
    HAB_AOI=/app/data/aoi/lakes.geojson \
    HAB_CUBES=/data/cubes \
    HAB_CHECKPOINT=/models/best.ckpt

# ---------------------------------------------------------------- serving target --
FROM base AS serve
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1
CMD ["uvicorn", "serving.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ----------------------------------------------------------- torch-free serving --
# The same API serving an ONNX export (scripts/export_onnx.py) through onnxruntime, with
# no torch in the image: torch's ~750 MB was the floor of the `serve` target above. Its
# own base, not `FROM base`, because that stage installs torch first. The core
# dependencies are read from pyproject.toml minus torch/lightning rather than listed here
# a second time, so the two cannot drift apart.
#
#   python scripts/export_onnx.py --checkpoint models/best.ckpt --out models/onnx/best
#   docker build --target serve-onnx -t bloom-watch:onnx .
#   docker run -p 8000:8000 -v $PWD/models:/models -e HAB_CHECKPOINT=/models/onnx/best bloom-watch:onnx
FROM python:3.12-slim AS serve-onnx
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        libexpat1 libgomp1 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/habforecast/__init__.py src/habforecast/__init__.py
COPY scripts/serve_onnx_requirements.py scripts/
# The last command fails the build if anything pulled torch in after all.
RUN python scripts/serve_onnx_requirements.py > /tmp/requirements.txt \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip install --no-cache-dir --no-deps -e . \
    && ! python -c "import torch" 2>/dev/null
COPY src/ src/
COPY serving/ serving/
COPY scripts/ scripts/
COPY data/aoi/ data/aoi/
COPY data/reference/ data/reference/
# Served at /model-card; the API description links to it.
COPY docs/MODEL_CARD.md docs/MODEL_CARD.md
RUN useradd --create-home --uid 10001 hab && chown -R hab:hab /app
USER hab
ENV PYTHONPATH=/app/src \
    HAB_AOI=/app/data/aoi/lakes.geojson \
    HAB_CUBES=/data/cubes \
    HAB_CHECKPOINT=/models/onnx/best \
    HAB_PROB_CHECKPOINT=/models/onnx/ens_met
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1
CMD ["uvicorn", "serving.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ----------------------------------------------------------------- ingest target --
FROM base AS ingest
ENTRYPOINT ["python", "pipelines/ingest.py"]
CMD ["--start", "2018-01-01", "--end", "2024-12-31", "--workers", "12"]

# ------------------------------------------------------------------- tools target --
# Tests, linting and the offline renderers (case-study imagery, dashboard). Separate from
# the GPU training image on purpose: `make test` used to target that image, which installs
# [train,monitor] and therefore has no pytest -- the target failed with "executable file
# not found" rather than with a test result. CPU torch is already in `base`, so this stage
# adds only the test and lint tooling.
FROM base AS tools
USER root
COPY tests/ tests/
COPY monitoring/ monitoring/
RUN pip install --no-cache-dir -e ".[ci]" && chown -R hab:hab /app
USER hab
ENTRYPOINT []
CMD ["pytest", "-m", "not network", "-q"]
