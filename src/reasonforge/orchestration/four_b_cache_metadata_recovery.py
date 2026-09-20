"""Append-only authorization for two exact HF bookkeeping files; no CUDA imports.

The model snapshot, v1 disk authorization, thresholds, and runtime stay frozen.
Only the diagnostic cache inventory changes: a missing-adapter marker and the
matching-revision main ref. Every live launch still compares the full inventory.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from reasonforge.checkpoints.hashing import atomic_write_json

from .four_b_disk_recovery import CODE_PATHS as DISK_CODE_PATHS
from .four_b_disk_recovery import RECOVERY as DISK_RECOVERY
from .four_b_disk_recovery import inventory_cache
from .four_b_transfer import (
    E36_REVISION,
    FourBIntegrityError,
    _binding,
    _load_json,
    _mapping,
    _verify_binding,
)

RECOVERY_ROOT = "evidence/training/E36/cache_metadata_recovery_v1"
RECOVERY = f"{RECOVERY_ROOT}/AUTHORIZATION.json"
ORIGINAL_CODE = f"{RECOVERY_ROOT}/ORIGINAL_four_b_disk_recovery.py"
VALIDATION = f"{RECOVERY_ROOT}/CPU_VALIDATION.json"
IDENTITY = "RFPT-E36-CACHE-METADATA-RECOVERY-v1"
CACHE = ".cache/huggingface/hub/models--Qwen--Qwen3-4B"
AUXILIARY = (
    f".no_exist/{E36_REVISION}/adapter_config.json",
    "refs/main",
    f"trees/{E36_REVISION}.json",
)
CODE_PATHS = (
    *DISK_CODE_PATHS,
    "src/reasonforge/orchestration/four_b_cache_metadata_recovery.py",
    "scripts/recover_e36_cache_metadata.py",
    "tests/test_four_b_cache_metadata_recovery.py",
)


def validate_inventory_delta(prior: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    """Permit only +40 logical bytes and +2 files; all snapshot fields stay exact."""
    expected = dict(prior)
    expected["unique_cache_file_content_bytes"] = int(prior["unique_cache_file_content_bytes"]) + 40
    expected["unique_cache_file_count"] = int(prior["unique_cache_file_count"]) + 2
    if dict(current) != expected:
        raise FourBIntegrityError("E36 metadata recovery changes more than approved bookkeeping")


def auxiliary_bindings(root: Path) -> dict[str, Any]:
    cache = root / CACHE
    actual = {
        path.relative_to(cache).as_posix()
        for path in cache.rglob("*")
        if path.is_file() and path.relative_to(cache).parts[0] not in {"snapshots", "blobs"}
    }
    if actual != set(AUXILIARY):
        raise FourBIntegrityError("E36 auxiliary cache file set is not the exact approved set")
    if any((cache / name).is_symlink() for name in AUXILIARY):
        raise FourBIntegrityError("E36 auxiliary cache links are not authorized")
    if (cache / AUXILIARY[0]).read_bytes() != b"":
        raise FourBIntegrityError("E36 adapter absence marker must be empty")
    if (cache / AUXILIARY[1]).read_bytes() != E36_REVISION.encode("ascii"):
        raise FourBIntegrityError("E36 main ref must equal the frozen revision")
    return {name: _binding(root, f"{CACHE}/{name}") for name in AUXILIARY}


def verify_metadata_recovery(
    root: Path,
    disk_recovery: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    if candidate is None and not (root / RECOVERY).exists():
        return None
    value = (
        _load_json(root / RECOVERY, "cache metadata recovery") if candidate is None else candidate
    )
    if (
        value.get("recovery_id") != IDENTITY
        or value.get("status") != "AUTHORIZED_FOR_BOUNDED_CUDA_QUALIFICATION"
        or value.get("scientific_recipe_preserved") is not True
        or value.get("cuda_process_launched") is not False
        or value.get("model_loaded") is not False
    ):
        raise FourBIntegrityError("E36 invalid cache metadata recovery identity")
    if (
        _verify_binding(root, value.get("disk_recovery"), "original disk recovery")["path"]
        != DISK_RECOVERY
    ):
        raise FourBIntegrityError("E36 cache metadata recovery binds a noncanonical disk record")
    original = _verify_binding(root, value.get("original_code"), "original disk verifier")
    frozen_code = _mapping(disk_recovery["code_inventory"], "original code")
    old_binding = _mapping(frozen_code[DISK_CODE_PATHS[0]], "original disk verifier")
    if original["path"] != ORIGINAL_CODE or any(
        original[key] != old_binding[key] for key in ("sha256", "bytes")
    ):
        raise FourBIntegrityError("E36 original verifier archive differs from v1 authorization")
    code = _mapping(value.get("code_inventory"), "metadata recovery code")
    if set(code) != set(CODE_PATHS):
        raise FourBIntegrityError("E36 metadata recovery code inventory is incomplete")
    for name, binding in code.items():
        if _verify_binding(root, binding, name)["path"] != name:
            raise FourBIntegrityError("E36 metadata recovery code path changed")
    validation_binding = _verify_binding(root, value.get("cpu_validation"), "CPU validation")
    if validation_binding["path"] != VALIDATION:
        raise FourBIntegrityError("E36 noncanonical cache metadata validation")
    validation = _load_json(root / VALIDATION, "metadata CPU validation")
    if (
        validation.get("status") != "PASSED"
        or validation.get("exit_codes") != [0, 0, 0]
        or validation.get("code_inventory") != dict(code)
        or validation.get("cuda_used") is not False
    ):
        raise FourBIntegrityError("E36 cache metadata validation did not qualify current code")
    _verify_binding(root, validation.get("log"), "metadata CPU validation log")
    preserved = _mapping(value.get("preserved_evidence"), "preserved E36 evidence")
    if not preserved:
        raise FourBIntegrityError("E36 cache metadata recovery omitted prior evidence")
    for name, binding in preserved.items():
        if _verify_binding(root, binding, name)["path"] != name:
            raise FourBIntegrityError("E36 preserved evidence path changed")
    validate_inventory_delta(
        _mapping(disk_recovery.get("inventory"), "original inventory"),
        _mapping(value.get("inventory"), "metadata inventory"),
    )
    if auxiliary_bindings(root) != value.get("auxiliary_files"):
        raise FourBIntegrityError("E36 approved auxiliary cache metadata changed")
    return dict(value)


def publish_metadata_recovery(root: Path) -> dict[str, Any]:
    """Publish once after CPU qualification, retaining every original evidence byte."""
    root = root.resolve()
    if (root / RECOVERY).exists():
        raise FourBIntegrityError("E36 cache metadata recovery already exists")
    disk = _load_json(root / DISK_RECOVERY, "original disk authorization")
    inventory = inventory_cache(root, root / CACHE, expected_weight_bytes=8044982000)
    validate_inventory_delta(_mapping(disk["inventory"], "original inventory"), inventory)
    code = {name: _binding(root, name) for name in CODE_PATHS}
    validation = _load_json(root / VALIDATION, "metadata CPU validation")
    if (
        validation.get("status") != "PASSED"
        or validation.get("exit_codes") != [0, 0, 0]
        or validation.get("code_inventory") != code
    ):
        raise FourBIntegrityError("E36 cache metadata CPU validation must pass before publication")
    preserved = {
        path.relative_to(root).as_posix(): _binding(root, path.relative_to(root).as_posix())
        for path in sorted((root / "evidence/training/E36").rglob("*"))
        if path.is_file() and root / RECOVERY_ROOT not in path.parents
    }
    value = {
        "schema_version": 1,
        "recovery_id": IDENTITY,
        "status": "AUTHORIZED_FOR_BOUNDED_CUDA_QUALIFICATION",
        "experiment_id": "E36",
        "disk_recovery": _binding(root, DISK_RECOVERY),
        "original_code": _binding(root, ORIGINAL_CODE),
        "code_inventory": code,
        "cpu_validation": _binding(root, VALIDATION),
        "preserved_evidence": preserved,
        "inventory": inventory,
        "auxiliary_files": auxiliary_bindings(root),
        "scientific_recipe_preserved": True,
        "cuda_process_launched": False,
        "model_loaded": False,
        "diagnosis": "Exact snapshot unchanged; HF optional adapter lookup added one empty "
        "negative-cache marker and one 40-byte ref equal to the frozen revision. "
        "Existing acquisition tree metadata is separately hashed here; its historical "
        "content hash was not recorded by disk recovery v1.",
    }
    verify_metadata_recovery(root, disk, candidate=value)
    atomic_write_json(root / RECOVERY, value)
    return value
