from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

REVISION = "b4be1225c9b593ffa79f2bf46d8a84a83d385c67"

CACHE = (
    ROOT
    / ".cache"
    / "upstream"
    / "morena-1.5b-instruct"
    / REVISION
)

CONFIG_PATH = CACHE / "config.json"
TOKENIZER_PATH = CACHE / "tokenizer.json"
MODEL_CODE_PATH = CACHE / "modeling_morena.py"
LOAD_EXAMPLE_PATH = CACHE / "load_example.py"

OUTPUT_PATH = ROOT / "reports" / "g1" / "model_architecture_inventory.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def classes(tree: ast.AST) -> list[str]:
    return [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
    ]


def functions(tree: ast.AST) -> list[str]:
    return [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def imports(tree: ast.AST) -> list[str]:
    values: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            values.update(alias.name for alias in node.names)

        if isinstance(node, ast.ImportFrom):
            values.add(node.module or "")

    return sorted(values)


def selected_literals(tree: ast.AST) -> list[str]:
    terms = (
        "reserved_",
        "rope",
        "attention",
        "flash",
        "sdpa",
        "cache",
        "gradient",
        "generate",
        "tokenizer",
        "safetensors",
        "cuda",
        "cpu",
        "dtype",
    )

    values = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and any(term.lower() in node.value.lower() for term in terms)
    }

    return sorted(values)


def tokenizer_summary(data: dict[str, Any]) -> dict[str, Any]:
    model = data.get("model", {})
    added = data.get("added_tokens", [])

    return {
        "model_type": model.get("type"),
        "vocabulary_entries": len(model.get("vocab", {})),
        "merge_count": len(model.get("merges", [])),
        "added_token_count": len(added),
        "added_tokens": [
            {
                "id": token.get("id"),
                "content": token.get("content"),
                "special": token.get("special"),
            }
            for token in added
        ],
        "normalizer": data.get("normalizer"),
        "pre_tokenizer": data.get("pre_tokenizer"),
        "post_processor": data.get("post_processor"),
        "decoder": data.get("decoder"),
    }


config = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
tokenizer = json.loads(TOKENIZER_PATH.read_text(encoding="utf-8-sig"))

model_source = MODEL_CODE_PATH.read_text(encoding="utf-8-sig")
loader_source = LOAD_EXAMPLE_PATH.read_text(encoding="utf-8-sig")

model_tree = ast.parse(model_source)
loader_tree = ast.parse(loader_source)

inventory = {
    "schema_version": "1.0",
    "upstream": {
        "repo": "vamboai/morena-1.5b-instruct",
        "revision": REVISION,
    },
    "source_hashes": {
        "config.json": sha256(CONFIG_PATH),
        "tokenizer.json": sha256(TOKENIZER_PATH),
        "modeling_morena.py": sha256(MODEL_CODE_PATH),
        "load_example.py": sha256(LOAD_EXAMPLE_PATH),
    },
    "config": config,
    "tokenizer": tokenizer_summary(tokenizer),
    "modeling_source": {
        "classes": classes(model_tree),
        "functions": functions(model_tree),
        "imports": imports(model_tree),
        "selected_literals": selected_literals(model_tree),
    },
    "reference_loader": {
        "classes": classes(loader_tree),
        "functions": functions(loader_tree),
        "imports": imports(loader_tree),
        "selected_literals": selected_literals(loader_tree),
    },
}

OUTPUT_PATH.write_text(
    json.dumps(inventory, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

model_config = config["model"]
train_config = config["train"]

assert model_config["d_model"] % model_config["n_head"] == 0
head_dim = model_config["d_model"] // model_config["n_head"]

print("G1_STATIC_ARCHAEOLOGY=PASS")
print(f"OUTPUT={OUTPUT_PATH.relative_to(ROOT)}")
print("CONFIG_LAYOUT=CUSTOM_MORENA")
print(f"VOCAB_SIZE={model_config['vocab_size']}")
print(f"HIDDEN_SIZE={model_config['d_model']}")
print(f"NUM_LAYERS={model_config['n_layer']}")
print(f"ATTENTION_HEADS={model_config['n_head']}")
print(f"KV_HEADS={model_config['n_kv_head']}")
print(f"HEAD_DIM={head_dim}")
print(f"FFN_SIZE={model_config['d_ff']}")
print(f"ROPE_THETA={model_config['rope_theta']}")
print(f"TIED_EMBEDDINGS={model_config['tie_embeddings']}")
print(f"SEQUENCE_LENGTH={train_config['seq_len']}")
print(
    "TOKENIZER_VOCAB_ENTRIES="
    f"{inventory['tokenizer']['vocabulary_entries']}"
)
print(
    "ADDED_TOKENS="
    f"{inventory['tokenizer']['added_token_count']}"
)
