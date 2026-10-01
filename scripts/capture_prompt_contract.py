from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REVISION = "b4be1225c9b593ffa79f2bf46d8a84a83d385c67"

SOURCE_PATH = (
    ROOT
    / ".cache"
    / "upstream"
    / "morena-1.5b-instruct"
    / REVISION
    / "load_example.py"
)

OUTPUT_PATH = ROOT / "reports" / "g1" / "prompt_source_inventory.json"


source = SOURCE_PATH.read_text(encoding="utf-8-sig")
tree = ast.parse(source)
lines = source.splitlines()


segments: list[dict[str, object]] = []

for node in ast.walk(tree):
    segment = ast.get_source_segment(source, node)

    valid_segment = isinstance(segment, str)
    contains_user_marker = valid_segment and "<reserved_0>" in segment
    contains_assistant_marker = valid_segment and "<reserved_1>" in segment

    marker_segment = contains_user_marker or contains_assistant_marker

    segments.extend(
        [
            {
                "node_type": type(node).__name__,
                "line_start": getattr(node, "lineno", None),
                "line_end": getattr(node, "end_lineno", None),
                "source": segment,
            }
        ]
        * int(marker_segment)
    )


generate_functions = [
    node
    for node in ast.walk(tree)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    and node.name == "generate"
]

assert len(generate_functions) == 1, (
    "Expected exactly one generate() function in pinned reference loader."
)

generate_node = generate_functions[0]

generate_source = "\n".join(
    lines[
        generate_node.lineno - 1 :
        generate_node.end_lineno
    ]
)


assert segments, (
    "No prompt-marker source segments were found."
)


payload = {
    "schema_version": "1.0",
    "repo": "vamboai/morena-1.5b-instruct",
    "revision": REVISION,
    "source_file": "load_example.py",
    "source_sha256": hashlib.sha256(
        SOURCE_PATH.read_bytes()
    ).hexdigest(),
    "reserved_marker_segments": segments,
    "generate_function": generate_source,
}


OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH.write_text(
    json.dumps(payload, indent=2, ensure_ascii=False),
    encoding="utf-8",
)


print("PROMPT_SOURCE_CAPTURE=PASS")
print(f"MARKER_SEGMENTS={len(segments)}")
print(f"OUTPUT={OUTPUT_PATH.relative_to(ROOT)}")
