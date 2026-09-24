#!/usr/bin/env bash
# Install the CUDA build of PyTorch into the active environment.
#
# Kept out of environment.yml on purpose. With `--extra-index-url`, pip treats PyPI as an
# equally valid source and picks the highest version number, which is the CPU-only wheel
# -- training then silently runs on CPU. Using `--index-url` makes the PyTorch index the
# only source, so the CUDA wheel is the only candidate.
#
# CUDA_TAG must match your driver. Check with `nvidia-smi`:
#   driver >= 525 -> cu121/cu124   driver >= 550 -> cu126/cu128   driver >= 580 -> cu130
set -euo pipefail

CUDA_TAG="${CUDA_TAG:-cu130}"
PYTHON="${PYTHON:-python}"

echo "==> Installing torch (${CUDA_TAG}) from the PyTorch index"
"$PYTHON" -m pip install --force-reinstall \
  --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
  torch torchvision

echo "==> Verifying"
"$PYTHON" - <<'PYEOF'
import sys, torch
print(f"torch {torch.__version__} | built for CUDA {torch.version.cuda}")
if not torch.cuda.is_available():
    print("ERROR: CUDA not available. Wrong CUDA_TAG for this driver, or a CPU wheel won.")
    sys.exit(1)
p = torch.cuda.get_device_properties(0)
print(f"OK: {p.name} | {p.total_memory/1e9:.2f} GB | sm_{p.major}{p.minor}")
PYEOF
