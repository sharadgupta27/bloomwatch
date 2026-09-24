#!/usr/bin/env bash
# Create the `hab` conda environment and install the project in editable mode.
set -euo pipefail
CONDA_EXE="${CONDA_EXE:-$HOME/miniforge3/Scripts/conda.exe}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> Creating conda env 'hab' from environment.yml"
"$CONDA_EXE" env create -f "$REPO_ROOT/environment.yml" --yes || \
  "$CONDA_EXE" env update -f "$REPO_ROOT/environment.yml" --prune

PY="$(dirname "$CONDA_EXE")/../envs/hab/python.exe"
echo "==> Installing habforecast (editable)"
"$PY" -m pip install -e "$REPO_ROOT[all]" --no-deps

echo "==> Verifying"
"$PY" - <<'PYEOF'
import torch
print(f"torch {torch.__version__} | cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"  {p.name} | {p.total_memory/1e9:.2f} GB | sm_{p.major}{p.minor}")
PYEOF
echo "==> Done. Activate with:  conda activate hab"
