from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))


from morena_r.contracts.evaluation import EvalInput  # noqa: E402
from morena_r.models.native_morena import (  # noqa: E402
    GenerationConfig,
    NativeMorenaRuntime,
    RuntimeConfig,
)
from morena_r.prompting.b0 import (  # noqa: E402
    render_b0_prompt,
)


payload = json.loads(
    sys.stdin.read()
)

eval_input = EvalInput.model_validate(
    payload["eval_input"]
)

generation_config = GenerationConfig(
    **payload["generation"]
)

runtime_config = RuntimeConfig(
    **payload["runtime"]
)

prompt = render_b0_prompt(
    eval_input
)

prompt_sha256 = hashlib.sha256(
    prompt.encode("utf-8")
).hexdigest()


runtime = NativeMorenaRuntime(
    root=ROOT,
    runtime_config=runtime_config,
)

generation = runtime.generate(
    prompt=prompt,
    config=generation_config,
)


result = {
    "schema_version": "1.0",
    "case_id": eval_input.case_id,
    "raw_output": generation.raw_output,
    "prompt_sha256": prompt_sha256,
    "input_token_count": (
        generation.input_token_count
    ),
    "output_token_count": (
        generation.output_token_count
    ),
    "stop_reason": (
        generation.stop_reason
    ),
    "model_inference_seconds": (
        generation.inference_seconds
    ),
}


sys.stdout.write(
    json.dumps(
        result,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
)
