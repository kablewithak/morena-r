from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]

REVISION = "b4be1225c9b593ffa79f2bf46d8a84a83d385c67"

MODEL_SOURCE_DIR = (
    ROOT
    / ".cache"
    / "upstream"
    / "morena-1.5b-instruct"
    / REVISION
)

sys.path.insert(0, str(MODEL_SOURCE_DIR))

import modeling_morena  # noqa: E402


assert hasattr(modeling_morena, "Transformer")
assert hasattr(modeling_morena, "ModelConfig")
assert hasattr(modeling_morena, "Attention")
assert hasattr(modeling_morena, "MLP")


print("MORENA_MODEL_IMPORT=PASS")
print(f"TORCH_VERSION={torch.__version__}")
print(f"CUDA_AVAILABLE={torch.cuda.is_available()}")
print(
    "TRANSFORMER_CLASS="
    f"{modeling_morena.Transformer.__module__}."
    f"{modeling_morena.Transformer.__name__}"
)
