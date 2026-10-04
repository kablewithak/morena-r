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


PILOT = (
    ROOT
    / "data"
    / "pilot"
    / "g2_development_pilot_v1.json"
)

OUTPUT = (
    ROOT
    / "reports"
    / "b0"
    / "integration_probe_v2.json"
)


def git_output(
    *args: str,
) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
    ).strip()


if git_output(
    "status",
    "--porcelain",
):
    raise RuntimeError(
        "B0 integration probe requires a clean Git worktree."
    )


if OUTPUT.exists():
    raise FileExistsError(
        f"Probe output already exists: {OUTPUT}"
    )


cases = load_eval_cases(
    PILOT
)

case = next(
    case
    for case in cases
    if (
        case.input.case_id
        == "ref-respond-supported"
    )
)

prompt = render_b0_prompt(
    case.input
)

prompt_sha256 = hashlib.sha256(
    prompt.encode("utf-8")
).hexdigest()

generation_config = GenerationConfig(
    seed=1234,
    temperature=0.7,
    top_p=0.9,
    max_new_tokens=48,
)


print("B0_NATIVE_MODEL_LOAD_START=TRUE")

runtime = NativeMorenaRuntime(
    root=ROOT
)

print("B0_NATIVE_MODEL_LOAD=PASS")
print(
    f"INPUT_CASE={case.input.case_id}"
)
print(
    f"PROMPT_SHA256={prompt_sha256}"
)
print(
    f"MAX_NEW_TOKENS={generation_config.max_new_tokens}"
)
print("B0_NATIVE_GENERATION_START=TRUE")


generation = runtime.generate(
    prompt=prompt,
    config=generation_config,
)

parsed = parse_model_response(
    generation.raw_output
)


report = {
    "schema_version": "1.0",
    "state": (
        "B0_NATIVE_ADAPTER_INTEGRATION_PROBE"
    ),
    "implementation_commit": git_output(
        "rev-parse",
        "HEAD",
    ),
    "git_dirty": False,
    "repository": REPOSITORY,
    "revision": REVISION,
    "case_id": case.input.case_id,
    "prompt_version": B0_PROMPT_VERSION,
    "prompt_sha256": prompt_sha256,
    "prompt": prompt,
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
        "valid": (
            parsed.response is not None
        ),
        "failure_code": (
            parsed.failure.code.value
            if parsed.failure is not None
            else None
        ),
    },
    "baseline_reproduced": False,
    "improvement_claimed": False,
}


OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT.write_text(
    json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
    ) + "\n",
    encoding="utf-8",
    newline="\n",
)


print("B0_NATIVE_INTEGRATION_PROBE=PASS")
print(
    "INPUT_TOKENS="
    f"{generation.input_token_count}"
)
print(
    "OUTPUT_TOKENS="
    f"{generation.output_token_count}"
)
print(
    "STOP_REASON="
    f"{generation.stop_reason}"
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
    "PARSE_VALID="
    f"{parsed.response is not None}"
)

if parsed.failure is not None:
    print(
        "PARSE_FAILURE="
        f"{parsed.failure.code.value}"
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
