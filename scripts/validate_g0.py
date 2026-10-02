from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

INSTRUCT_REVISION = "b4be1225c9b593ffa79f2bf46d8a84a83d385c67"
BASE_REVISION = "e98192f5f3ddb97d118e107ece37399f3cbe84d2"
GGUF_REVISION = "ed64de7262b6edbe1af14c340ed4ea368a36349d"

CACHE = (
    ROOT
    / ".cache"
    / "upstream"
    / "morena-1.5b-instruct"
    / INSTRUCT_REVISION
)

MODEL_IDENTITY = ROOT / "data" / "manifests" / "model_identity.json"
INTEGRITY_MANIFEST = (
    ROOT / "data" / "manifests" / "upstream_file_integrity_g0.json"
)
ENVIRONMENT_INVENTORY = ROOT / "reports" / "g0" / "environment_inventory.json"
HF_REFS = ROOT / "experiments" / "registry" / "g0_hf_refs.txt"

REQUIRED_CACHE_FILES = (
    "README.md",
    "config.json",
    "tokenizer.json",
    "modeling_morena.py",
    "load_example.py",
    "SHA256SUMS",
)

REQUIRED_TRACKED_ARTIFACTS = (
    MODEL_IDENTITY,
    INTEGRITY_MANIFEST,
    ENVIRONMENT_INVENTORY,
    HF_REFS,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    assert ROOT.joinpath(".git").exists(), "Git repository is missing."

    for filename in REQUIRED_CACHE_FILES:
        path = CACHE / filename
        assert path.is_file(), f"Missing pinned inspection file: {path}"

    for path in REQUIRED_TRACKED_ARTIFACTS:
        assert path.is_file(), f"Missing G0 artifact: {path}"

    identity = json.loads(MODEL_IDENTITY.read_text(encoding="utf-8-sig"))

    assert (
        identity["primary_model"]["revision"] == INSTRUCT_REVISION
    ), "Instruct revision mismatch."

    assert (
        identity["base_comparator"]["revision"] == BASE_REVISION
    ), "Base revision mismatch."

    assert (
        identity["upstream_gguf"]["revision"] == GGUF_REVISION
    ), "GGUF revision mismatch."

    integrity = json.loads(INTEGRITY_MANIFEST.read_text(encoding="utf-8-sig"))

    by_file = {record["file"]: record for record in integrity}

    required_verified = (
        "config.json",
        "tokenizer.json",
        "modeling_morena.py",
        "load_example.py",
    )

    for filename in required_verified:
        record = by_file[filename]
        path = CACHE / filename

        assert record["matches_upstream"] is True, (
            f"Upstream checksum did not match for {filename}"
        )

        assert sha256(path) == record["local_sha256"], (
            f"Local file changed after G0 capture: {filename}"
        )

    readme = by_file["README.md"]

    assert readme["local_sha256"] == sha256(CACHE / "README.md"), (
        "README.md changed after G0 capture."
    )

    assert readme["upstream_sha256"] is None, (
        "README checksum state differs from captured G0 evidence."
    )

    print("G0_VALIDATION=PASS")
    print(f"INSTRUCT_REVISION={INSTRUCT_REVISION}")
    print(f"BASE_REVISION={BASE_REVISION}")
    print(f"GGUF_REVISION={GGUF_REVISION}")
    print("UPSTREAM_CHECKSUM_MATCHES=4")
    print("README_UPSTREAM_CHECKSUM=NOT_AVAILABLE")
    print(
        "BASELINE_REPRODUCED="
        + str(identity["claims"]["baseline_reproduced"]).upper()
    )
    print(
        "INFERENCE_VERIFIED="
        + str(identity["claims"]["inference_verified"]).upper()
    )
    print(
        "IMPROVEMENT_CLAIMED="
        + str(identity["claims"]["improvement_claimed"]).upper()
    )


if __name__ == "__main__":
    main()
