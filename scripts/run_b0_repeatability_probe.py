from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))


from morena_r.evaluation.dataset import (  # noqa: E402
    load_eval_cases,
)
from morena_r.evaluation.parser import (  # noqa: E402
    parse_model_response,
)
from morena_r.models.native_morena import (  # noqa: E402
    GenerationConfig,
    NativeMorenaRuntime,
    REPOSITORY,
    REVISION,
)
from morena_r.prompting.b0 import (  # noqa: E402
    B0_PROMPT_VERSION,
    render_b0_prompt,
)


FIRST_REPORT = (
    ROOT
    / "reports"
    / "b0"
    / "integration_probe_v2.json"
)

OUTPUT = (
    ROOT
    / "reports"
    / "b0"
    / "repeatability_probe_v1.json"
)

PILOT = (
    ROOT
    / "data"
    / "pilot"
    / "g2_development_pilot_v1.json"
)

RUNTIME_PATH = (
    "src/morena_r/models/native_morena.py"
)

PROMPT_PATH = (
    "src/morena_r/prompting/b0.py"
)


def git_output(
    *args: str,
) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
    ).strip()


def git_bytes(
    revision: str,
    path: str,
) -> bytes:
    return subprocess.check_output(
        [
            "git",
            "show",
            f"{revision}:{path}",
        ],
        cwd=ROOT,
    )


def sha256_bytes(
    value: bytes,
) -> str:
    return hashlib.sha256(
        value
    ).hexdigest()


if git_output(
    "status",
    "--porcelain",
):
    raise RuntimeError(
        "B0 repeatability probe requires a clean Git worktree."
    )


if not FIRST_REPORT.is_file():
    raise FileNotFoundError(
        "First B0 integration probe is missing."
    )


if OUTPUT.exists():
    raise FileExistsError(
        f"Repeatability output already exists: {OUTPUT}"
    )


first = json.loads(
    FIRST_REPORT.read_text(
        encoding="utf-8"
    )
)

first_commit = first[
    "implementation_commit"
]

current_commit = git_output(
    "rev-parse",
    "HEAD",
)


runtime_first_hash = sha256_bytes(
    git_bytes(
        first_commit,
        RUNTIME_PATH,
    )
)

runtime_current_hash = sha256_bytes(
    (ROOT / RUNTIME_PATH).read_bytes()
)

prompt_first_hash = sha256_bytes(
    git_bytes(
        first_commit,
        PROMPT_PATH,
    )
)

prompt_current_hash = sha256_bytes(
    (ROOT / PROMPT_PATH).read_bytes()
)


if (
    runtime_first_hash
    != runtime_current_hash
):
    raise RuntimeError(
        "Native runtime changed since first probe."
    )


if (
    prompt_first_hash
    != prompt_current_hash
):
    raise RuntimeError(
        "B0 prompt compiler changed since first probe."
    )


cases = load_eval_cases(
    PILOT
)

case = next(
    case
    for case in cases
    if (
        case.input.case_id
        == first["case_id"]
    )
)

prompt = render_b0_prompt(
    case.input
)

prompt_sha256 = hashlib.sha256(
    prompt.encode("utf-8")
).hexdigest()


if (
    prompt_sha256
    != first["prompt_sha256"]
):
    raise RuntimeError(
        "Rendered prompt differs from first probe."
    )


generation_config = GenerationConfig(
    seed=first["generation"]["seed"],
    temperature=(
        first["generation"]["temperature"]
    ),
    top_p=(
        first["generation"]["top_p"]
    ),
    max_new_tokens=(
        first["generation"][
            "max_new_tokens"
        ]
    ),
)


print("B0_REPEAT_MODEL_LOAD_START=TRUE")

runtime = NativeMorenaRuntime(
    root=ROOT
)

print("B0_REPEAT_MODEL_LOAD=PASS")
print("B0_REPEAT_GENERATION_START=TRUE")


generation = runtime.generate(
    prompt=prompt,
    config=generation_config,
)

parsed = parse_model_response(
    generation.raw_output
)


generated_ids_match = (
    list(generation.generated_ids)
    == first["generation"][
        "generated_ids"
    ]
)

raw_output_match = (
    generation.raw_output
    == first["generation"][
        "raw_output"
    ]
)

stop_reason_match = (
    generation.stop_reason
    == first["generation"][
        "stop_reason"
    ]
)

parse_valid = (
    parsed.response is not None
)

parse_valid_match = (
    parse_valid
    == first["parse"]["valid"]
)

failure_code = (
    parsed.failure.code.value
    if parsed.failure is not None
    else None
)

failure_code_match = (
    failure_code
    == first["parse"][
        "failure_code"
    ]
)


report = {
    "schema_version": "1.0",
    "state": (
        "B0_NATIVE_REPEATABILITY_PROBE"
    ),
    "first_implementation_commit": (
        first_commit
    ),
    "repeat_implementation_commit": (
        current_commit
    ),
    "git_dirty": False,
    "repository": REPOSITORY,
    "revision": REVISION,
    "case_id": case.input.case_id,
    "prompt_version": B0_PROMPT_VERSION,
    "prompt_sha256": prompt_sha256,
    "runtime_source_sha256": (
        runtime_current_hash
    ),
    "prompt_source_sha256": (
        prompt_current_hash
    ),
    "runtime_source_unchanged": True,
    "prompt_source_unchanged": True,
    "generation": {
        "seed": generation_config.seed,
        "temperature": (
            generation_config.temperature
        ),
        "top_p": generation_config.top_p,
        "max_new_tokens": (
            generation_config.max_new_tokens
        ),
        "input_token_count": (
            generation.input_token_count
        ),
        "output_token_count": (
            generation.output_token_count
        ),
        "generated_ids": list(
            generation.generated_ids
        ),
        "raw_output": (
            generation.raw_output
        ),
        "stop_reason": (
            generation.stop_reason
        ),
        "inference_seconds": (
            generation.inference_seconds
        ),
        "output_tokens_per_second": (
            generation.output_tokens_per_second
        ),
    },
    "parse": {
        "valid": parse_valid,
        "failure_code": failure_code,
    },
    "comparison": {
        "generated_ids_match": (
            generated_ids_match
        ),
        "raw_output_match": (
            raw_output_match
        ),
        "stop_reason_match": (
            stop_reason_match
        ),
        "parse_valid_match": (
            parse_valid_match
        ),
        "failure_code_match": (
            failure_code_match
        ),
    },
    "baseline_reproduced": False,
    "improvement_claimed": False,
}


OUTPUT.write_text(
    json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
    ) + "\n",
    encoding="utf-8",
    newline="\n",
)


print("B0_REPEATABILITY_PROBE=PASS")
print(
    "RUNTIME_SOURCE_UNCHANGED=TRUE"
)
print(
    "PROMPT_SOURCE_UNCHANGED=TRUE"
)
print(
    "GENERATED_IDS_MATCH="
    f"{generated_ids_match}"
)
print(
    "RAW_OUTPUT_MATCH="
    f"{raw_output_match}"
)
print(
    "STOP_REASON_MATCH="
    f"{stop_reason_match}"
)
print(
    "PARSE_VALID_MATCH="
    f"{parse_valid_match}"
)
print(
    "FAILURE_CODE_MATCH="
    f"{failure_code_match}"
)
print(
    "INFERENCE_SECONDS="
    f"{generation.inference_seconds:.3f}"
)
print(
    "OUTPUT_TOKENS_PER_SECOND="
    f"{generation.output_tokens_per_second:.6f}"
)
print(
    "RAW_OUTPUT="
    f"{generation.raw_output!r}"
)
print(
    "BASELINE_REPRODUCED=FALSE"
)
print(
    "IMPROVEMENT_CLAIMED=FALSE"
)
print(
    "REPORT="
    f"{OUTPUT.relative_to(ROOT)}"
)
