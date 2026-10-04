from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from pydantic import TypeAdapter


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))

from morena_r.contracts.actions import (  # noqa: E402
    ModelResponse,
    ToolCall,
    ToolResult,
)
from morena_r.contracts.evaluation import (  # noqa: E402
    AttemptObservation,
    AttemptRecord,
    EvalCase,
    EvalGold,
    EvalInput,
    RunSummary,
)
from morena_r.contracts.scoring import ScoreRecord  # noqa: E402
from morena_r.evaluation.tool_episode import (  # noqa: E402
    EpisodeEvent,
    ToolEpisodeResult,
)


schemas = {
    "score_record.schema.json": ScoreRecord.model_json_schema(),
    "model_response.schema.json": TypeAdapter(
        ModelResponse
    ).json_schema(),
    "tool_call.schema.json": TypeAdapter(
        ToolCall
    ).json_schema(),
    "tool_result.schema.json": ToolResult.model_json_schema(),
    "eval_input.schema.json": EvalInput.model_json_schema(),
    "eval_gold.schema.json": EvalGold.model_json_schema(),
    "eval_case.schema.json": EvalCase.model_json_schema(),
    "attempt_record.schema.json": AttemptRecord.model_json_schema(),
    "attempt_observation.schema.json": AttemptObservation.model_json_schema(),
    "run_summary.schema.json": RunSummary.model_json_schema(),
    "episode_event.schema.json": EpisodeEvent.model_json_schema(),
    "tool_episode_result.schema.json": ToolEpisodeResult.model_json_schema(),
}


output_dir = ROOT / "contracts"
output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


for filename, schema in schemas.items():
    serialized = json.dumps(
        schema,
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    ) + "\n"

    output = output_dir / filename

    output.write_text(
        serialized,
        encoding="utf-8",
        newline="\n",
    )

    digest = hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()

    print(
        f"SCHEMA={filename} SHA256={digest}"
    )


print("CONTRACT_SCHEMA_EXPORT=PASS")
