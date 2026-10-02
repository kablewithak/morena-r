from __future__ import annotations

import hashlib
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))


from morena_r.contracts.actions import (  # noqa: E402
    RecordFixture,
    ToolName,
    ToolResult,
)
from morena_r.contracts.evaluation import (  # noqa: E402
    EvalInput,
    EvalMessage,
    ParseFailure,
)
from morena_r.evaluation.dataset import (  # noqa: E402
    dataset_sha256,
    load_eval_cases,
)
from morena_r.evaluation.tool_episode import (  # noqa: E402
    BoundedToolEpisodeRunner,
    EpisodeTerminalStatus,
)
from morena_r.reporting.run_bundle import (  # noqa: E402
    REQUIRED_BUNDLE_FILES,
)
from morena_r.tools.simulator import (  # noqa: E402
    FixtureState,
    ToolPermissions,
)


BUNDLE = (
    ROOT
    / "reports"
    / "g2"
    / "reference_pilot_v1"
)

PILOT = (
    ROOT
    / "data"
    / "pilot"
    / "g2_development_pilot_v1.json"
)


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


actual_files = {
    path.name
    for path in BUNDLE.iterdir()
    if path.is_file()
}

assert actual_files == set(
    REQUIRED_BUNDLE_FILES
)


for line in (
    BUNDLE
    / "checksums.sha256"
).read_text(
    encoding="utf-8"
).splitlines():
    digest, filename = line.split(
        "  ",
        1,
    )

    assert (
        sha256_file(
            BUNDLE / filename
        )
        == digest
    )


manifest = json.loads(
    (
        BUNDLE
        / "manifest.json"
    ).read_text(
        encoding="utf-8"
    )
)

cases = load_eval_cases(
    PILOT
)

assert manifest["git_dirty"] is False
assert (
    manifest["dataset_hash"]
    == dataset_sha256(cases)
)


metrics = json.loads(
    (
        BUNDLE
        / "metrics.json"
    ).read_text(
        encoding="utf-8"
    )
)

summary = metrics["summary"]

assert summary["expected_cases"] == 8
assert summary["attempted_cases"] == 8
assert summary["completed"] == 5
assert summary["invalid_responses"] == 1
assert summary["timeouts"] == 1
assert summary["execution_errors"] == 1


metric_by_id = {
    row["metric_id"]: row
    for row in metrics["metrics"]
}

assert (
    metric_by_id[
        "action_correct"
    ]["eligible_instances"]
    == 8
)

assert (
    metric_by_id[
        "action_correct"
    ]["estimate"]
    == 0.625
)

assert (
    metric_by_id[
        "answerability_correct"
    ]["eligible_instances"]
    == 8
)

assert (
    metric_by_id[
        "answerability_correct"
    ]["estimate"]
    == 0.625
)

assert (
    metric_by_id[
        "schema_valid"
    ]["eligible_instances"]
    == 6
)

assert (
    metric_by_id[
        "tool_identity_correct"
    ]["eligible_instances"]
    == 2
)

assert (
    metric_by_id[
        "tool_identity_correct"
    ]["estimate"]
    == 0.5
)


failure_rows = [
    json.loads(line)
    for line in (
        BUNDLE
        / "failures.jsonl"
    ).read_text(
        encoding="utf-8"
    ).splitlines()
    if line
]

failure_by_case = {
    row["case_id"]: row
    for row in failure_rows
}

assert set(failure_by_case) == {
    "ref-invalid-json",
    "ref-timeout",
    "ref-context-overflow",
    "ref-wrong-tool",
}

assert (
    failure_by_case[
        "ref-invalid-json"
    ]["parse_error"]
    == "PARSE_ERROR"
)

assert (
    failure_by_case[
        "ref-timeout"
    ]["runtime_error"]
    == "TIMEOUT"
)

assert (
    failure_by_case[
        "ref-context-overflow"
    ]["runtime_error"]
    == "CONTEXT_OVERFLOW"
)

assert (
    failure_by_case[
        "ref-wrong-tool"
    ]["failure_labels"]
    == ["wrong_tool"]
)


class GateAdapter:
    def generate(
        self,
        eval_input: EvalInput,
        tool_results: tuple[ToolResult, ...],
    ) -> str:
        if not tool_results:
            return (
                '{"decision":"CALL_TOOL",'
                '"answerability":"NOT_APPLICABLE",'
                '"tool_call":{'
                '"tool":"get_record",'
                '"arguments":{"record_id":"gate-record"}}}'
            )

        assert tool_results[-1].success is True

        return (
            '{"decision":"RESPOND",'
            '"answerability":"SUPPORTED",'
            '"answer":"Gate fixture retrieved."}'
        )

    def repair(
        self,
        eval_input: EvalInput,
        *,
        raw_output: str,
        parse_failure: ParseFailure,
        tool_results: tuple[ToolResult, ...],
    ) -> str:
        raise AssertionError(
            "G2 gate round-trip should not require repair."
        )


gate_input = EvalInput(
    case_id="g2-gate-tool-roundtrip",
    family_id="g2-gate-tool-roundtrip",
    languages=("en",),
    messages=(
        EvalMessage(
            role="user",
            content="Retrieve the gate fixture.",
        ),
    ),
    evidence=(),
    available_tools=(
        ToolName.GET_RECORD,
    ),
    permitted_tools=(
        ToolName.GET_RECORD,
    ),
    evaluation_time_utc=datetime(
        2026,
        1,
        1,
        tzinfo=timezone.utc,
    ),
)

gate_state = FixtureState(
    permissions=ToolPermissions(
        get_record=True,
    ),
    records=(
        RecordFixture(
            record_id="gate-record",
            text="Synthetic G2 gate record.",
            observed_at=date(
                2026,
                1,
                1,
            ),
        ),
    ),
)

episode = BoundedToolEpisodeRunner(
    adapter=GateAdapter(),
).run(
    eval_input=gate_input,
    fixture_state=gate_state,
)

assert (
    episode.status
    == EpisodeTerminalStatus.COMPLETED
)

assert len(episode.turns) == 2
assert len(episode.tool_results) == 1
assert episode.tool_results[0].success is True


print("G2_VALIDATION=PASS")
print(
    "REFERENCE_DATASET_SHA256="
    f"{manifest['dataset_hash']}"
)
print(
    "REFERENCE_IMPLEMENTATION_COMMIT="
    f"{manifest['implementation_commit']}"
)
print("REFERENCE_CASES=8")
print("REFERENCE_ACCOUNTING_RECONCILED=TRUE")
print("DETERMINISTIC_REPORTING=TRUE")
print("BOUNDED_TOOL_ROUNDTRIP=PASS")
print("FORMAT_REPAIR_BOUND=1")
print("BASELINE_REPRODUCED=FALSE")
print("IMPROVEMENT_CLAIMED=FALSE")
