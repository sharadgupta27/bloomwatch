"""Shared pytest configuration.

`src` is put on the path here so the suite runs from a clean checkout without an
editable install -- CI installs the package properly, but a contributor cloning and
running `pytest` should not have to.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# `monitoring` is a top-level package outside src/ (it is an ops concern, not part of the
# installable library), so the repo root also needs to be importable.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The predictor's native stack (torch, pyproj, pandas/xarray) must be imported on the
# main thread before any test drives the API. FastAPI's TestClient runs the app in a
# worker thread, and on Windows the first import of that combination from a non-main
# thread corrupts the process -- pytest dies with "access violation" at event-loop close
# and no test failure. The full suite happened to be safe because earlier test modules
# import these at collection; running test_api.py on its own was not. The app imports
# the same module at startup for the same reason (see serving/api/main.py).
# torch first, explicitly: the predictor module stopped importing it (the ONNX serving
# path has none), so importing the predictor no longer brings it in, and torch loaded
# after pyproj/pandas crashed the suite the same way.
import torch  # noqa: E402, F401

import habforecast.inference.predictor  # noqa: E402, F401


@pytest.fixture(autouse=True)
def _no_eumetsat_credentials(monkeypatch, tmp_path_factory):
    """Hide any real ~/.eumdac credentials: with them, `data.olci.default_reader` would
    send a test's live path to the EUMETSAT Data Store over the network."""
    monkeypatch.setenv("EUMDAC_CONFIG_DIR", str(tmp_path_factory.mktemp("no_eumdac")))
    # Keep per-lake water masks a test computes out of the real data/cache, and keep the
    # API from warming the model and outlook in a background thread during a test.
    monkeypatch.setenv("HAB_STATIC_CACHE", str(tmp_path_factory.mktemp("statics")))
    monkeypatch.setenv("HAB_WARMUP", "0")
    monkeypatch.setenv("HAB_OUTLOOK_CACHE", str(tmp_path_factory.mktemp("outlook") / "o.json"))
