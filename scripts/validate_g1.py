from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

IDENTITY = ROOT / "data" / "manifests" / "model_identity.json"
WEIGHTS = ROOT / "reports" / "g1" / "weights_integrity.json"
STATE_DICT = ROOT / "reports" / "g1" / "state_dict_compatibility.json"
INFERENCE = ROOT / "reports" / "g1" / "inference_smoke.json"


def load(path: Path) -> dict:
    assert path.is_file(), f"Missing required G1 artifact: {path}"
    return json.loads(path.read_text(encoding="utf-8-sig"))


identity = load(IDENTITY)
weights = load(WEIGHTS)
state_dict = load(STATE_DICT)
inference = load(INFERENCE)


assert identity["status"] == "G1_INFERENCE_VERIFIED"

assert identity["primary_model"]["weights_downloaded"] is True
assert identity["primary_model"]["weights_verified"] is True

assert identity["claims"]["tokenizer_verified"] is True
assert identity["claims"]["prompt_format_verified"] is True
assert identity["claims"]["inference_verified"] is True

assert identity["claims"]["baseline_reproduced"] is False
assert identity["claims"]["improvement_claimed"] is False


assert weights["checksum_match"] is True
assert weights["state"] == "DOWNLOADED_AND_SHA256_VERIFIED"

assert state_dict["strict_load"] is True
assert state_dict["state"] == "STRICT_STATE_DICT_LOAD_VERIFIED"
assert state_dict["missing_keys"] == []
assert state_dict["unexpected_keys"] == []
assert state_dict["shape_mismatches"] == {}


assert inference["state"] == "CPU_REFERENCE_SMOKE_EXECUTED"
assert inference["qualified_device"] == "cpu"
assert inference["attention_mode"] == "sdpa"

assert inference["eos_id"] == 2
assert inference["user_marker_id"] == 3
assert inference["assistant_marker_id"] == 4

assert inference["input_ids"][0] == 2
assert inference["generated_ids"]

assert inference["revision"] == identity["primary_model"]["revision"]
assert inference["weight_sha256"] == weights["actual_sha256"]

assert (
    state_dict["weight_sha256"]
    == weights["actual_sha256"]
    == inference["weight_sha256"]
)


print("G1_VALIDATION=PASS")
print(f"PARAMETER_COUNT={state_dict['parameter_count']}")
print(f"CHECKPOINT_TENSORS={weights['tensor_count']}")
print(f"INFERENCE_DEVICE={inference['qualified_device']}")
print(f"INFERENCE_SECONDS={inference['inference_seconds']}")
print(f"SAMPLED_TOKEN_ID={inference['sampled_token_id']}")
print("BASELINE_REPRODUCED=FALSE")
print("IMPROVEMENT_CLAIMED=FALSE")
