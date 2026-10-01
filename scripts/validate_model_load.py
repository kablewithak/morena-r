from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]

REVISION = "b4be1225c9b593ffa79f2bf46d8a84a83d385c67"
ATTENTION_MODE = "sdpa"

UPSTREAM = (
    ROOT
    / ".cache"
    / "upstream"
    / "morena-1.5b-instruct"
    / REVISION
)

CONFIG_PATH = UPSTREAM / "config.json"
MODEL_SOURCE = UPSTREAM / "modeling_morena.py"
WEIGHTS_PATH = UPSTREAM / "model.safetensors"

WEIGHT_RECEIPT = ROOT / "reports" / "g1" / "weights_integrity.json"

OUTPUT_PATH = (
    ROOT
    / "reports"
    / "g1"
    / "state_dict_compatibility.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


assert CONFIG_PATH.is_file()
assert MODEL_SOURCE.is_file()
assert WEIGHTS_PATH.is_file()
assert WEIGHT_RECEIPT.is_file()


receipt = json.loads(
    WEIGHT_RECEIPT.read_text(encoding="utf-8-sig")
)

assert receipt["checksum_match"] is True
assert receipt["state"] == "DOWNLOADED_AND_SHA256_VERIFIED"
assert receipt["actual_sha256"] == sha256(WEIGHTS_PATH)


sys.path.insert(0, str(UPSTREAM))

spec = importlib.util.spec_from_file_location(
    "modeling_morena",
    MODEL_SOURCE,
)

assert spec is not None
assert spec.loader is not None

modeling_morena = importlib.util.module_from_spec(spec)

sys.modules["modeling_morena"] = modeling_morena
spec.loader.exec_module(modeling_morena)


config = json.loads(
    CONFIG_PATH.read_text(encoding="utf-8-sig")
)

model_config = modeling_morena.ModelConfig(
    **config["model"]
)


print(
    "MODEL_CONFIG_SIGNATURE="
    + str(inspect.signature(modeling_morena.ModelConfig))
)

print(
    "TRANSFORMER_SIGNATURE="
    + str(inspect.signature(modeling_morena.Transformer))
)


with torch.device("meta"):
    model = modeling_morena.Transformer(
        model_config,
        ATTENTION_MODE,
    )


model_state = model.state_dict()

model_keys = set(model_state)


with safe_open(
    WEIGHTS_PATH,
    framework="pt",
    device="cpu",
) as handle:
    weight_keys = set(handle.keys())

    weight_shapes = {
        key: tuple(handle.get_slice(key).get_shape())
        for key in handle.keys()
    }


model_shapes = {
    key: tuple(tensor.shape)
    for key, tensor in model_state.items()
}


missing_keys = sorted(model_keys - weight_keys)
unexpected_keys = sorted(weight_keys - model_keys)

shape_mismatches = {
    key: {
        "model": list(model_shapes[key]),
        "checkpoint": list(weight_shapes[key]),
    }
    for key in sorted(model_keys & weight_keys)
    if model_shapes[key] != weight_shapes[key]
}


assert not missing_keys, (
    f"Checkpoint missing model tensors: {missing_keys}"
)

assert not unexpected_keys, (
    f"Checkpoint contains unexpected tensors: {unexpected_keys}"
)

assert not shape_mismatches, (
    f"Tensor shape mismatches: {shape_mismatches}"
)


parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)


print("STATE_DICT_NAMES=PASS")
print("STATE_DICT_SHAPES=PASS")
print(f"MODEL_STATE_TENSORS={len(model_keys)}")
print(f"CHECKPOINT_TENSORS={len(weight_keys)}")
print(f"PARAMETER_COUNT={parameter_count}")
print("LOADING_CHECKPOINT=TRUE")


state_dict = load_file(
    WEIGHTS_PATH,
    device="cpu",
)


load_result = model.load_state_dict(
    state_dict,
    strict=True,
    assign=True,
)


assert len(load_result.missing_keys) == 0
assert len(load_result.unexpected_keys) == 0


meta_parameters = [
    name
    for name, parameter in model.named_parameters()
    if parameter.is_meta
]

assert not meta_parameters, (
    f"Parameters remained on meta device: {meta_parameters}"
)


loaded_parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)


assert loaded_parameter_count == parameter_count


parameter_dtypes = sorted(
    {
        str(parameter.dtype)
        for parameter in model.parameters()
    }
)


report = {
    "schema_version": "1.0",
    "repository": "vamboai/morena-1.5b-instruct",
    "revision": REVISION,
    "weight_sha256": receipt["actual_sha256"],
    "checkpoint_tensor_count": len(weight_keys),
    "model_state_tensor_count": len(model_keys),
    "missing_keys": missing_keys,
    "unexpected_keys": unexpected_keys,
    "shape_mismatches": shape_mismatches,
    "parameter_count": parameter_count,
    "loaded_parameter_count": loaded_parameter_count,
    "parameter_dtypes": parameter_dtypes,
    "attention_mode": ATTENTION_MODE,
    "strict_load": True,
    "state": "STRICT_STATE_DICT_LOAD_VERIFIED",
}


OUTPUT_PATH.write_text(
    json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


print("STRICT_STATE_DICT_LOAD=PASS")
print(f"PARAMETER_DTYPES={parameter_dtypes}")
print(f"REPORT={OUTPUT_PATH.relative_to(ROOT)}")
