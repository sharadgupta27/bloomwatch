"""Print the torch-free serving requirements: core + `serve-onnx` extras, minus torch.

Read from pyproject.toml at image build time (docker/Dockerfile, target `serve-onnx`) so
the dependency list exists in one place. Only torch and lightning are dropped -- the
two packages the ONNX backend (inference.onnx_backend) exists to do without.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

DROP = ("torch", "lightning")


def requirements(pyproject: Path = Path("pyproject.toml")) -> list[str]:
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
    extras = project["optional-dependencies"]
    reqs = list(project["dependencies"]) + list(extras["serve"])
    reqs += [r for r in extras["serve-onnx"] if not r.startswith("habforecast")]
    return [r for r in reqs if re.split(r"[<>=!~\[ ;]", r, maxsplit=1)[0] not in DROP]


if __name__ == "__main__":
    print("\n".join(requirements()))
