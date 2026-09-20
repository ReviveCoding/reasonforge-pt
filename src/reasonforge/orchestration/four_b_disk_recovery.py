"""Versioned post-acquisition disk accounting; no model or CUDA imports.

The frozen materialization estimate includes model weights. Credit only the exact
frozen weight footprint, after independently verifying the pinned cache. Metadata,
filesystem overhead, runtime growth and checkpoint retention receive no credit.
"""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from reasonforge.checkpoints.hashing import atomic_write_bytes, atomic_write_json, sha256_file

from .four_b_transfer import (
    E36_CONFIG,
    E36_DISPOSITION,
    E36_PREFLIGHT,
    E36_REVISION,
    DiskUsageReader,
    FourBIntegrityError,
    _binding,
    _config,
    _load_json,
    _mapping,
    _safe_path,
    _verify_binding,
    build_e36_preflight,
)

RECOVERY_ROOT = "evidence/training/E36/disk_recovery_v1"
RECOVERY = f"{RECOVERY_ROOT}/POST_ACQUISITION_AUTHORIZATION.json"
ORIGINAL_PREFLIGHT = f"{RECOVERY_ROOT}/ORIGINAL_FOUR_B_PREFLIGHT.json"
CLEANUP_AUDIT = "evidence/compute_budget/E36_DISK_CLEANUP_AUDIT.md"
VALIDATION = f"{RECOVERY_ROOT}/CPU_VALIDATION.json"
CONFIG_SHA256 = "0b8299c0b5d50fef5ad25427619448b5a0ac2bf2db7e58694c7831a84f0d1a54"
CODE_PATHS = (
    "src/reasonforge/orchestration/four_b_disk_recovery.py",
    "src/reasonforge/orchestration/four_b_transfer.py",
    "scripts/recover_e36_post_acquisition_disk.py",
    "tests/test_four_b_disk_recovery.py",
    "tests/test_four_b_transfer.py",
    "tests/test_four_b_execution.py",
    "tests/test_four_b_cuda_worker.py",
)
EXPECTED_FILES = {
    ".gitattributes",
    "config.json",
    "generation_config.json",
    "LICENSE",
    "merges.txt",
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
    "model.safetensors.index.json",
    "README.md",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
}


def inventory_cache(root: Path, cache: Path, *, expected_weight_bytes: int) -> dict[str, Any]:
    """Hash each exact-revision snapshot file once and verify its HF blob identity.

    Unique content bytes are logical lengths deduplicated by filesystem identity.
    They are deliberately not described as allocated clusters/physical disk use.
    Current free space is obtained separately from the filesystem itself.
    """
    snapshot = cache / "snapshots" / E36_REVISION
    if not snapshot.is_dir() or {p.name for p in snapshot.iterdir()} != EXPECTED_FILES:
        raise FourBIntegrityError("E36 pinned snapshot file set is incomplete or unexpected")
    records: dict[str, Any] = {}
    unique: dict[tuple[int, int], int] = {}
    for path in sorted(snapshot.iterdir()):
        target = path.resolve(strict=True)
        if target.parent != (cache / "blobs").resolve() or not path.is_symlink():
            raise FourBIntegrityError("E36 recovery requires canonical in-cache snapshot links")
        stat = target.stat()
        if stat.st_dev != root.stat().st_dev or not stat.st_ino:
            raise FourBIntegrityError("E36 cache filesystem identity cannot be established")
        digest = sha256_file(target)
        blob_id = target.name
        if len(blob_id) == 64:
            valid = digest == blob_id
        elif len(blob_id) == 40:
            git_hash = hashlib.sha1(usedforsecurity=False)
            git_hash.update(f"blob {stat.st_size}\0".encode())
            with target.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    git_hash.update(chunk)
            valid = git_hash.hexdigest() == blob_id
        else:
            valid = False
        after = target.stat()
        if not valid or (after.st_size, after.st_mtime_ns, after.st_ino, after.st_dev) != (
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ino,
            stat.st_dev,
        ):
            raise FourBIntegrityError(f"E36 cache blob digest changed or is invalid: {path.name}")
        unique[(stat.st_dev, stat.st_ino)] = stat.st_size
        records[path.name] = {
            "path": path.relative_to(root).as_posix(),
            "target": target.relative_to(root).as_posix(),
            "sha256": digest,
            "bytes": stat.st_size,
            "blob_id": blob_id,
        }
    index = _load_json(snapshot / "model.safetensors.index.json", "weight index")
    weight_map = _mapping(index.get("weight_map"), "weight map")
    shards = {str(value) for value in weight_map.values()}
    if not weight_map or shards != {
        name for name in EXPECTED_FILES if name.endswith(".safetensors")
    }:
        raise FourBIntegrityError("E36 weight index does not cover exactly the frozen shards")
    weights = sum(int(records[name]["bytes"]) for name in shards)
    tensor_bytes = _mapping(index.get("metadata"), "weight index metadata").get("total_size")
    if weights != expected_weight_bytes or not isinstance(tensor_bytes, int):
        raise FourBIntegrityError("E36 acquired weight size does not match frozen model")
    if not 0 < tensor_bytes < weights:
        raise FourBIntegrityError("E36 weight index tensor bytes are invalid")
    # Include unrelated cache contents in the diagnostic total, never in the credit.
    for path in cache.rglob("*"):
        if path.is_file():
            target = path.resolve(strict=True)
            if cache.resolve() not in target.parents:
                raise FourBIntegrityError("E36 cache link escapes its canonical directory")
            stat = target.stat()
            unique[(stat.st_dev, stat.st_ino)] = stat.st_size
    return {
        "snapshot_revision": E36_REVISION,
        "files": records,
        "snapshot_content_bytes": sum(int(row["bytes"]) for row in records.values()),
        "unique_cache_file_content_bytes": sum(unique.values()),
        "unique_cache_file_count": len(unique),
        "allocated_disk_bytes": None,
        "allocated_disk_bytes_note": "Logical file lengths deduplicated by device/inode; "
        "filesystem allocation and link overhead are not credited.",
        "verified_weight_bytes": weights,
        "tensor_bytes_from_index": tensor_bytes,
    }


def _accounting(root: Path, disk_usage_reader: DiskUsageReader) -> dict[str, Any]:
    config, digest = _config(root)
    if digest != CONFIG_SHA256:
        raise FourBIntegrityError("E36 disk recovery requires the exact frozen configuration")
    policy = _mapping(config.get("resource_policy"), "resource policy")
    inventory = inventory_cache(
        root,
        _safe_path(root, str(policy["canonical_cache_path"])),
        expected_weight_bytes=int(_mapping(config.get("model"), "model")["weight_bytes"]),
    )
    preflight = build_e36_preflight(root, disk_usage_reader=disk_usage_reader)
    credit = int(inventory["verified_weight_bytes"])
    disk = preflight["disk"]
    remaining = int(disk["required_free_bytes"]) - credit
    if credit > int(disk["estimated_model_and_runtime_bytes"]):
        raise FourBIntegrityError("E36 model credit exceeds the frozen materialization estimate")
    if not preflight["hardware"]["static_identity_matches"]:
        raise FourBIntegrityError("E36 recovery cannot bypass hardware identity qualification")
    if disk["free_bytes"] < remaining:
        raise FourBIntegrityError("INSUFFICIENT_DISK_FOR_FROZEN_REMAINING_RECIPE")
    return {
        "inventory": inventory,
        "static_preflight": preflight,
        "disk": {
            **disk,
            "capacity_sufficient": True,
            "acquired_weight_credit_bytes": credit,
            "remaining_required_free_bytes": remaining,
            "remaining_runtime_bytes": int(disk["estimated_model_and_runtime_bytes"]) - credit,
            "headroom_after_remaining_budget_bytes": int(disk["free_bytes"]) - remaining,
            "accounting_version": "RFPT-E36-POST-ACQUISITION-DISK-v1",
        },
    }


def publish_disk_recovery(root: Path) -> dict[str, Any]:
    """Preserve prior evidence and publish a content-bound ready preflight once."""
    root = root.resolve()
    if (root / E36_DISPOSITION).exists() or (root / RECOVERY).exists():
        raise FourBIntegrityError("E36 disk recovery or final disposition already exists")
    if any((root / "artifacts/training/E36").rglob("RUN_IDENTITY.json")):
        raise FourBIntegrityError("E36 disk recovery must precede scientific run identities")
    validation = _load_json(root / VALIDATION, "CPU recovery validation")
    if validation.get("status") != "PASSED" or validation.get("exit_codes") != [0, 0, 0]:
        raise FourBIntegrityError("E36 recovery CPU validation did not pass")
    _verify_binding(root, validation.get("log"), "CPU validation log")
    if validation.get("code_hashes") != {path: sha256_file(root / path) for path in CODE_PATHS}:
        raise FourBIntegrityError("E36 recovery code changed after CPU validation")
    accounting = _accounting(root, shutil.disk_usage)
    prior = (root / E36_PREFLIGHT).read_bytes()
    archive = root / ORIGINAL_PREFLIGHT
    if archive.exists() and archive.read_bytes() != prior:
        raise FourBIntegrityError("E36 original preflight archive conflicts")
    atomic_write_bytes(archive, prior)
    value = {
        "schema_version": 1,
        "status": "AUTHORIZED_FOR_BOUNDED_CUDA_QUALIFICATION",
        "recovery_id": "RFPT-E36-POST-ACQUISITION-DISK-v1",
        "experiment_id": "E36",
        "configuration": _binding(root, E36_CONFIG),
        "original_preflight": _binding(root, ORIGINAL_PREFLIGHT),
        "cleanup_audit": _binding(root, CLEANUP_AUDIT),
        "cpu_validation": _binding(root, VALIDATION),
        "code_inventory": {path: _binding(root, path) for path in CODE_PATHS},
        "model_loaded": False,
        "cuda_process_launched": False,
        "scientific_recipe_preserved": True,
        "substitute_experiment_run": False,
        "historical_preacquisition_pass_is_not_a_durable_artifact": True,
        **accounting,
    }
    atomic_write_json(root / RECOVERY, value)
    recovered = dict(accounting["static_preflight"])
    recovered.update(
        status="READY_FOR_BOUNDED_CUDA_QUALIFICATION",
        disk=accounting["disk"],
        post_acquisition_recovery=_binding(root, RECOVERY),
    )
    atomic_write_json(root / E36_PREFLIGHT, recovered)
    verify_disk_recovery(root, recovered, live=True)
    return value


def verify_disk_recovery(root: Path, preflight: Mapping[str, Any], *, live: bool) -> None:
    """Verify durable recovery; launch verification also hashes cache and measures disk."""
    from .four_b_cache_metadata_recovery import verify_metadata_recovery

    binding = _verify_binding(root, preflight.get("post_acquisition_recovery"), "disk recovery")
    if binding["path"] != RECOVERY:
        raise FourBIntegrityError("E36 noncanonical disk recovery")
    value = _load_json(root / RECOVERY, "disk recovery")
    if (
        value.get("recovery_id") != "RFPT-E36-POST-ACQUISITION-DISK-v1"
        or value.get("status") != "AUTHORIZED_FOR_BOUNDED_CUDA_QUALIFICATION"
    ):
        raise FourBIntegrityError("E36 invalid disk recovery identity")
    for key in ("configuration", "original_preflight", "cleanup_audit", "cpu_validation"):
        _verify_binding(root, value.get(key), key)
    code = _mapping(value.get("code_inventory"), "recovery code")
    metadata_recovery = verify_metadata_recovery(root, value)
    if set(code) != set(CODE_PATHS):
        raise FourBIntegrityError("E36 recovery code inventory is incomplete")
    for key, raw in code.items():
        if metadata_recovery is not None and key == CODE_PATHS[0]:
            # The additive authorization verifies this original source from its
            # immutable archive and binds the updated verifier independently.
            continue
        if _verify_binding(root, raw, key)["path"] != key:
            raise FourBIntegrityError("E36 recovery code path changed")
    static = dict(_mapping(value.get("static_preflight"), "original static result"))
    static.update(
        status="READY_FOR_BOUNDED_CUDA_QUALIFICATION",
        disk=value.get("disk"),
        post_acquisition_recovery=binding,
    )
    if dict(preflight) != static:
        raise FourBIntegrityError("E36 recovered preflight is not canonical")
    if live:
        observed = _accounting(root, shutil.disk_usage)
        expected_inventory = (
            metadata_recovery["inventory"]
            if metadata_recovery is not None
            else value.get("inventory")
        )
        if observed["inventory"] != expected_inventory:
            raise FourBIntegrityError("E36 cache changed after disk recovery")
        for key in (
            "required_free_bytes",
            "remaining_required_free_bytes",
            "acquired_weight_credit_bytes",
            "safety_reserve_bytes",
        ):
            if observed["disk"][key] != value["disk"][key]:
                raise FourBIntegrityError("E36 frozen disk accounting changed")
