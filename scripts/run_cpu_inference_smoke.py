from __future__ import annotations

import gc
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]

REPOSITORY = "vamboai/morena-1.5b-instruct"
REVISION = "b4be1225c9b593ffa79f2bf46d8a84a83d385c67"

REFERENCE_LOADER_SHA256 = (
    "b0730bd0f6afea78ccb9151c340f0216eacf6335245d24957edae91316bae285"
)

ATTENTION_MODE = "sdpa"
DEVICE = "cpu"

SEED = 1234
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_NEW_TOKENS = 1

USER_MARKER = "<reserved_0>"
ASSISTANT_MARKER = "<reserved_1>"

USER_TEXT = "Ndeipi guta guru reZimbabwe?"

PROMPT = (
    f"{USER_MARKER}\n"
    f"{USER_TEXT}\n"
    f"{ASSISTANT_MARKER}\n"
)

UPSTREAM = (
    ROOT
    / ".cache"
    / "upstream"
    / "morena-1.5b-instruct"
    / REVISION
)

CONFIG_PATH = UPSTREAM / "config.json"
TOKENIZER_PATH = UPSTREAM / "tokenizer.json"
WEIGHTS_PATH = UPSTREAM / "model.safetensors"
MODEL_SOURCE = UPSTREAM / "modeling_morena.py"
REFERENCE_LOADER = UPSTREAM / "load_example.py"

WEIGHT_RECEIPT_PATH = (
    ROOT / "reports" / "g1" / "weights_integrity.json"
)

COMPATIBILITY_RECEIPT_PATH = (
    ROOT / "reports" / "g1" / "state_dict_compatibility.json"
)

OUTPUT_PATH = (
    ROOT / "reports" / "g1" / "inference_smoke.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


for required_path in (
    CONFIG_PATH,
    TOKENIZER_PATH,
    WEIGHTS_PATH,
    MODEL_SOURCE,
    REFERENCE_LOADER,
    WEIGHT_RECEIPT_PATH,
    COMPATIBILITY_RECEIPT_PATH,
):
    assert required_path.is_file(), (
        f"Required artifact missing: {required_path}"
    )


assert sha256(REFERENCE_LOADER) == REFERENCE_LOADER_SHA256, (
    "Pinned reference loader identity changed."
)


weight_receipt = json.loads(
    WEIGHT_RECEIPT_PATH.read_text(encoding="utf-8-sig")
)

compatibility_receipt = json.loads(
    COMPATIBILITY_RECEIPT_PATH.read_text(encoding="utf-8-sig")
)


assert weight_receipt["checksum_match"] is True
assert weight_receipt["state"] == "DOWNLOADED_AND_SHA256_VERIFIED"

assert compatibility_receipt["strict_load"] is True
assert compatibility_receipt["state"] == "STRICT_STATE_DICT_LOAD_VERIFIED"
assert compatibility_receipt["attention_mode"] == ATTENTION_MODE

assert sha256(WEIGHTS_PATH) == weight_receipt["actual_sha256"], (
    "Local model weights no longer match qualified SHA-256."
)


tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

EOS = tokenizer.token_to_id("<eos>")
USER_ID = tokenizer.token_to_id(USER_MARKER)
ASSISTANT_ID = tokenizer.token_to_id(ASSISTANT_MARKER)

assert EOS == 2
assert USER_ID == 3
assert ASSISTANT_ID == 4


prompt_body_ids = tokenizer.encode(
    PROMPT,
    add_special_tokens=False,
).ids

input_ids = [EOS] + prompt_body_ids


assert input_ids[0] == EOS
assert USER_ID in input_ids
assert ASSISTANT_ID in input_ids


print("PROMPT_CONTRACT=PASS")
print(f"EOS_ID={EOS}")
print(f"USER_MARKER_ID={USER_ID}")
print(f"ASSISTANT_MARKER_ID={ASSISTANT_ID}")
print(f"INPUT_TOKEN_COUNT={len(input_ids)}")
print(f"INPUT_IDS={input_ids}")
print(f"PROMPT={PROMPT!r}")


sys.path.insert(0, str(UPSTREAM))

import modeling_morena as M  # noqa: E402


config = json.loads(
    CONFIG_PATH.read_text(encoding="utf-8-sig")
)


load_started = time.perf_counter()

with torch.device("meta"):
    model = M.Transformer(
        M.ModelConfig(**config["model"]),
        ATTENTION_MODE,
    )


state_dict = load_file(
    WEIGHTS_PATH,
    device=DEVICE,
)


load_result = model.load_state_dict(
    state_dict,
    strict=True,
    assign=True,
)


assert len(load_result.missing_keys) == 0
assert len(load_result.unexpected_keys) == 0

del state_dict
gc.collect()

model.eval()

parameter_dtypes = sorted(
    {
        str(parameter.dtype)
        for parameter in model.parameters()
    }
)

parameter_devices = sorted(
    {
        str(parameter.device)
        for parameter in model.parameters()
    }
)

assert parameter_dtypes == ["torch.bfloat16"]
assert parameter_devices == ["cpu"]


load_seconds = time.perf_counter() - load_started


x = torch.tensor(
    [input_ids],
    dtype=torch.long,
    device=DEVICE,
)

torch.manual_seed(SEED)


print("MODEL_LOAD=PASS")
print(f"MODEL_LOAD_SECONDS={load_seconds:.3f}")
print(f"PARAMETER_DTYPES={parameter_dtypes}")
print(f"PARAMETER_DEVICES={parameter_devices}")
print("FORWARD_PASS_STARTING=TRUE")


inference_started = time.perf_counter()

with torch.inference_mode():
    positions = torch.arange(
        x.shape[1],
        device=x.device,
    ).unsqueeze(0)

    logits = model(
        x,
        positions,
        None,
        x.shape[1],
        None,
    )[:, -1, :].float()

    probabilities = torch.softmax(
        logits / max(TEMPERATURE, 1e-5),
        dim=-1,
    )

    sorted_probabilities, sorted_indices = torch.sort(
        probabilities,
        descending=True,
        dim=-1,
    )

    sorted_probabilities[
        sorted_probabilities.cumsum(-1)
        - sorted_probabilities
        > TOP_P
    ] = 0.0

    next_token = sorted_indices.gather(
        -1,
        torch.multinomial(
            sorted_probabilities
            / sorted_probabilities.sum(
                -1,
                keepdim=True,
            ),
            1,
        ),
    )


inference_seconds = time.perf_counter() - inference_started

sampled_token_id = int(next_token[0, 0])

generated_ids = (
    []
    if sampled_token_id == EOS
    else [sampled_token_id]
)

decoded_output = tokenizer.decode(
    generated_ids,
    skip_special_tokens=True,
)

sampled_token_piece = tokenizer.id_to_token(
    sampled_token_id
)

stop_reason = (
    "eos"
    if sampled_token_id == EOS
    else "max_new_tokens"
)


report = {
    "schema_version": "1.0",
    "repository": REPOSITORY,
    "revision": REVISION,
    "weight_sha256": weight_receipt["actual_sha256"],
    "reference_loader_sha256": REFERENCE_LOADER_SHA256,
    "runtime_profile": "windows_cpu_smoke",
    "upstream_reference_device": "cuda",
    "qualified_device": DEVICE,
    "device_adaptation": (
        "Upstream CUDA placement changed to CPU only. "
        "Model architecture, checkpoint, attention mode, tokenizer, "
        "prompt framing, leading EOS behavior and sampling rule retained."
    ),
    "torch_version": torch.__version__,
    "python_version": platform.python_version(),
    "platform": platform.platform(),
    "torch_threads": torch.get_num_threads(),
    "attention_mode": ATTENTION_MODE,
    "parameter_dtypes": parameter_dtypes,
    "seed": SEED,
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "max_new_tokens": MAX_NEW_TOKENS,
    "user_marker": USER_MARKER,
    "user_marker_id": USER_ID,
    "assistant_marker": ASSISTANT_MARKER,
    "assistant_marker_id": ASSISTANT_ID,
    "eos_id": EOS,
    "user_text": USER_TEXT,
    "prompt": PROMPT,
    "prompt_body_ids": prompt_body_ids,
    "input_ids": input_ids,
    "input_token_count": len(input_ids),
    "sampled_token_id": sampled_token_id,
    "sampled_token_piece": sampled_token_piece,
    "generated_ids": generated_ids,
    "decoded_output": decoded_output,
    "stop_reason": stop_reason,
    "model_load_seconds": load_seconds,
    "inference_seconds": inference_seconds,
    "sampled_tokens_per_second": (
        1.0 / inference_seconds
    ),
    "state": "CPU_REFERENCE_SMOKE_EXECUTED",
}


OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PATH.write_text(
    json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


print("G1_CPU_INFERENCE_SMOKE=PASS")
print(f"SAMPLED_TOKEN_ID={sampled_token_id}")
print(f"SAMPLED_TOKEN_PIECE={sampled_token_piece!r}")
print(f"DECODED_OUTPUT={decoded_output!r}")
print(f"STOP_REASON={stop_reason}")
print(f"INFERENCE_SECONDS={inference_seconds:.3f}")
print(
    "SAMPLED_TOKENS_PER_SECOND="
    f"{1.0 / inference_seconds:.6f}"
)
print(f"REPORT={OUTPUT_PATH.relative_to(ROOT)}")
