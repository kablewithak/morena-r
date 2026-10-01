from __future__ import annotations

import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]

REPO_ID = "vamboai/morena-1.5b-instruct"
REVISION = "b4be1225c9b593ffa79f2bf46d8a84a83d385c67"
FILENAME = "model.safetensors"

CACHE_DIR = (
    ROOT
    / ".cache"
    / "upstream"
    / "morena-1.5b-instruct"
    / REVISION
)

CHECKSUM_PATH = CACHE_DIR / "SHA256SUMS"

REPORT_PATH = (
    ROOT
    / "reports"
    / "g1"
    / "weights_integrity.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def upstream_sha256() -> str:
    rows = [
        line.strip().split(maxsplit=1)
        for line in CHECKSUM_PATH.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if line.strip()
    ]

    matches = [
        row[0].lower()
        for row in rows
        if len(row) == 2
        and row[1].lstrip("*") == FILENAME
    ]

    assert len(matches) == 1, (
        "Expected exactly one model.safetensors entry "
        "in pinned SHA256SUMS."
    )

    return matches[0]


CACHE_DIR.mkdir(parents=True, exist_ok=True)

expected_sha256 = upstream_sha256()

print(f"REPO={REPO_ID}")
print(f"REVISION={REVISION}")
print(f"FILE={FILENAME}")
print(f"EXPECTED_SHA256={expected_sha256}")
print("DOWNLOAD_STARTING=TRUE")

downloaded = Path(
    hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        revision=REVISION,
        local_dir=CACHE_DIR,
    )
)

assert downloaded.is_file(), (
    f"Downloaded artifact not found: {downloaded}"
)

assert downloaded.name == FILENAME

actual_sha256 = sha256(downloaded)

assert actual_sha256 == expected_sha256, (
    "Pinned weight SHA-256 mismatch."
)

with safe_open(
    downloaded,
    framework="pt",
    device="cpu",
) as handle:
    keys = list(handle.keys())
    metadata = handle.metadata()

    tensor_shapes = {
        key: list(handle.get_slice(key).get_shape())
        for key in keys
    }

assert keys, "Safetensors file contains no tensors."

report = {
    "schema_version": "1.0",
    "repository": REPO_ID,
    "revision": REVISION,
    "filename": FILENAME,
    "bytes": downloaded.stat().st_size,
    "expected_sha256": expected_sha256,
    "actual_sha256": actual_sha256,
    "checksum_match": True,
    "tensor_count": len(keys),
    "tensor_keys": keys,
    "tensor_shapes": tensor_shapes,
    "metadata": metadata,
    "state": "DOWNLOADED_AND_SHA256_VERIFIED",
}

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

REPORT_PATH.write_text(
    json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print("WEIGHT_DOWNLOAD=PASS")
print("WEIGHT_SHA256=PASS")
print(f"BYTES={downloaded.stat().st_size}")
print(f"TENSOR_COUNT={len(keys)}")
print(f"FIRST_TENSOR={keys[0]}")
print(f"LAST_TENSOR={keys[-1]}")
print(f"REPORT={REPORT_PATH.relative_to(ROOT)}")
