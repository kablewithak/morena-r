from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from morena_r.contracts.evaluation import EvalInput
from morena_r.evaluation.dataset import (
    dataset_sha256,
    load_eval_cases,
)
from morena_r.evaluation.runner import (
    EvaluationRunConfig,
    EvaluationRunner,
    ModelContextOverflowError,
    ModelTimeoutError,
)
from morena_r.reporting.run_bundle import (
    write_run_bundle,
)


ROOT = Path(__file__).resolve().parents[1]

PILOT_PATH = (
    ROOT
    / "data"
    / "pilot"
    / "g2_development_pilot_v1.json"
)

MODEL_IDENTITY_PATH = (
    ROOT
    / "data"
    / "manifests"
    / "model_identity.json"
)

ENVIRONMENT_LOCK_PATH = (
    ROOT
    / "requirements"
    / "g2-environment.lock.txt"
)

OUTPUT_DIR = (
    ROOT
    / "reports"
    / "g2"
    / "reference_pilot_v1"
)


class ScriptedReferenceAdapter:
    def generate(
        self,
        eval_input: EvalInput,
    ) -> str:
        outputs = {
            "ref-respond-supported": (
                '{"decision":"RESPOND",'
                '"answerability":"SUPPORTED",'
                '"answer":"Harare"}'
            ),
            "ref-ask-clarification": (
                '{"decision":"ASK_CLARIFICATION",'
                '"answerability":"INSUFFICIENT_CONTEXT",'
                '"question":"Which subject should I check?"}'
            ),
            "ref-refuse": (
                '{"decision":"REFUSE",'
                '"answerability":"NOT_APPLICABLE",'
                '"reason":"The synthetic action is disallowed."}'
            ),
            "ref-tool-correct": (
                '{"decision":"CALL_TOOL",'
                '"answerability":"NOT_APPLICABLE",'
                '"tool_call":{'
                '"tool":"get_record",'
                '"arguments":{"record_id":"rec-001"}}}'
            ),
            "ref-invalid-json": (
                '{"decision":'
            ),
            "ref-wrong-tool": (
                '{"decision":"CALL_TOOL",'
                '"answerability":"NOT_APPLICABLE",'
                '"tool_call":{'
                '"tool":"lookup_status",'
                '"arguments":{"subject_id":"case-001"}}}'
            ),
        }

        if (
            eval_input.case_id
            == "ref-timeout"
        ):
            raise ModelTimeoutError(
                "scripted reference timeout"
            )

        if (
            eval_input.case_id
            == "ref-context-overflow"
        ):
            raise ModelContextOverflowError(
                "scripted reference context overflow"
            )

        return outputs[
            eval_input.case_id
        ]


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def git_output(
    *args: str,
) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
    ).strip()


cases = load_eval_cases(
    PILOT_PATH
)

dataset_hash = dataset_sha256(
    cases
)

implementation_commit = git_output(
    "rev-parse",
    "HEAD",
)

git_status = git_output(
    "status",
    "--porcelain",
)

if git_status:
    raise RuntimeError(
        "G2 reference qualification requires a clean Git worktree."
    )

model_identity_hash = sha256_file(
    MODEL_IDENTITY_PATH
)

environment_lock_hash = sha256_file(
    ENVIRONMENT_LOCK_PATH
)

run_id = (
    "g2-reference-pilot-v1-"
    + dataset_hash[:12]
)

result = EvaluationRunner(
    config=EvaluationRunConfig(
        run_id=run_id,
        model_identity_hash=(
            model_identity_hash
        ),
        dataset_hash=dataset_hash,
    ),
    adapter=ScriptedReferenceAdapter(),
).run(cases)


assert result.summary.expected_cases == 8
assert result.summary.attempted_cases == 8
assert result.summary.completed == 5
assert result.summary.invalid_responses == 1
assert result.summary.timeouts == 1
assert result.summary.execution_errors == 1
assert result.summary.score_rows == 32


bundle_receipt_sha256 = write_run_bundle(
    output_dir=OUTPUT_DIR,
    cases=cases,
    result=result,
    implementation_commit=(
        implementation_commit
    ),
    git_dirty=False,
    environment_lock_sha256=(
        environment_lock_hash
    ),
    bundle_kind=(
        "g2_harness_reference_qualification"
    ),
    timing_mode=(
        "synthetic_reference_zero"
    ),
)


print("G2_REFERENCE_PILOT=PASS")
print(f"RUN_ID={run_id}")
print(f"DATASET_SHA256={dataset_hash}")
print(f"CASE_COUNT={len(cases)}")
print(
    f"COMPLETED={result.summary.completed}"
)
print(
    "INVALID_RESPONSES="
    f"{result.summary.invalid_responses}"
)
print(
    f"TIMEOUTS={result.summary.timeouts}"
)
print(
    "EXECUTION_ERRORS="
    f"{result.summary.execution_errors}"
)
print(
    "BUNDLE_CHECKSUM_RECEIPT_SHA256="
    f"{bundle_receipt_sha256}"
)
print(
    "OUTPUT="
    f"{OUTPUT_DIR.relative_to(ROOT)}"
)
