"""Evidence-bound administrative closure of underspecified P1 E36; no CUDA imports.

This distinct disposition never asserts execution or measured compute failure.
The original scientific config and original disposition verifier remain intact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from reasonforge.checkpoints.hashing import atomic_write_json, canonical_json_bytes, sha256_file
from reasonforge.config import load_registry

from .four_b_transfer import (
    E36_CONFIG,
    E36_DISPOSITION,
    E36_MODEL,
    E36_PREFLIGHT,
    E36_PROTOCOL,
    E36_REVISION,
    FourBIntegrityError,
    _binding,
    _config,
    _frozen_recipe,
    _load_json,
    _mapping,
    _safe_path,
    _verify_binding,
    _verify_preflight,
)

STATUS = "BLOCKED_PROTOCOL_UNDERSPECIFIED"
POLICY_ID = "RFPT-E36-PROTOCOL-BLOCK-DISPOSITION-v1"
ROOT = "evidence/training/E36/protocol_block_disposition_v1"
VALIDATION = f"{ROOT}/CPU_VALIDATION.json"
AUDIT = "evidence/training/E36/runtime_gate_audit_v1/EVIDENCE_BINDINGS.json"
AUDIT_SHA256 = "ee494b04150f8b871914926a8e7dfdab4994a81fea5f2b7dc72d62a8c5593c06"
REPORT = "evidence/training/E36/runtime_gate_audit_v1/BLOCKER.md"
REPORT_SHA256 = "3f6e41a0c2e728d31f32c6ed2c022eb5e0dc96f99f0a31097ae37806e42b2248"
SUCCESS = "evidence/training/E36/host_memory_recovery_v1/20260911T003528745358Z"
PROFILE_ROOT = "evidence/training/E36/runtime_qualification"
MODEL_CONFIG = f".cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/{E36_REVISION}/config.json"
ADAPTER = "artifacts/training/E10/E10-main-sft-seed1710-v1/final_adapter/adapter_model.safetensors"
CODE_PATHS = (
    "src/reasonforge/orchestration/four_b_protocol_disposition.py",
    "src/reasonforge/checkpoints/milestone_evidence.py",
    "scripts/finalize_e36_protocol_block.py",
    "tests/test_four_b_protocol_disposition.py",
    "tests/test_cp80_milestone_evidence.py",
    "docs/E36_PROTOCOL_BLOCK_DISPOSITION.md",
)


def _audit(root: Path) -> dict[str, Any]:
    for name, digest in ((AUDIT, AUDIT_SHA256), (REPORT, REPORT_SHA256)):
        if sha256_file(_safe_path(root, name)) != digest:
            raise FourBIntegrityError("E36 protocol blocker audit changed")
    audit = _load_json(root / AUDIT, "protocol blocker audit")
    for name, digest in _mapping(audit.get("source_sha256"), "audit source hashes").items():
        if sha256_file(_safe_path(root, name)) != digest:
            raise FourBIntegrityError(f"E36 protocol blocker source changed: {name}")
    for name in audit["missing_canonical_paths"]:
        if _safe_path(root, name).exists():
            raise FourBIntegrityError("E36 protocol blocker audit no longer matches runtime state")
    model = _load_json(root / MODEL_CONFIG, "pinned scale model config")
    with (root / ADAPTER).open("rb") as stream:
        size = int.from_bytes(stream.read(8), "little")
        if not 0 < size <= 1024**2:
            raise FourBIntegrityError("E36 starting adapter header size is invalid")
        header = json.loads(stream.read(size))
    name = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
    shape = _mapping(header.get(name), "starting adapter tensor").get("shape")
    if shape != [8, 2048] or model.get("hidden_size") != 2560:
        raise FourBIntegrityError("E36 audited adapter/model width mismatch changed")
    return {
        "audit": _binding(root, AUDIT),
        "report": _binding(root, REPORT),
        "adapter_tensor": name,
        "adapter_shape": shape,
        "scale_hidden_size": model["hidden_size"],
        "scale_model_config": _binding(root, MODEL_CONFIG),
        "incompatible_frozen_starting_adapter_verified": True,
    }


def _profiles(root: Path, ladder: list[Any]) -> dict[str, Any]:
    results = {}
    for step in ladder:
        name = f"{PROFILE_ROOT}/{step}.json"
        value = _load_json(root / name, "bounded profile")
        outcome = _mapping(value.get("outcome"), "bounded profile outcome")
        if (
            value.get("status") != "QUALIFIED"
            or value.get("qualification_scope") != "REAL_CUDA_QWEN3_4B_C0_PROFILE"
            or value.get("experiment_id") != "E36"
            or value.get("protocol_id") != E36_PROTOCOL
            or value.get("model_id") != E36_MODEL
            or value.get("model_revision") != E36_REVISION
            or value.get("fallback_step") != step
            or value.get("configuration") != _binding(root, E36_CONFIG)
            or value.get("preflight") != _binding(root, E36_PREFLIGHT)
            or value.get("single_model_instance") is not True
            or value.get("single_gpu_process") is not True
            or value.get("protected_outputs_accessed") is not False
            or outcome.get("cuda_available") is not True
            or outcome.get("cuda_device_name") != "NVIDIA GeForce RTX 4090 Laptop GPU"
            or outcome.get("model_instances_created") != 1
            or outcome.get("child_model_processes_created") != 0
            or outcome.get("optimizer_created") is not False
            or outcome.get("completed_optimizer_steps") != 0
            or outcome.get("final_checkpoint_completion") is not None
            or outcome.get("resumed_from") is not None
            or outcome.get("protected_outputs_accessed") is not False
            or not isinstance(outcome.get("peak_allocated_bytes"), int)
            or outcome["peak_allocated_bytes"] <= 0
        ):
            raise FourBIntegrityError(f"E36 bounded profile is invalid: {step}")
        results[step] = _binding(root, name)
    terminal = _load_json(root / SUCCESS / "TERMINAL.json", "successful recorder terminal")
    if terminal.get("status") != "COMPLETE" or terminal.get("returncode") != 0:
        raise FourBIntegrityError("E36 successful recorder terminal is invalid")
    logs = _mapping(terminal.get("raw_logs"), "successful recorder logs")
    if set(logs) != {"stdout.log", "stderr.log", "HOST_SAMPLES.jsonl"}:
        raise FourBIntegrityError("E36 successful recorder log inventory is incomplete")
    for name, digest in logs.items():
        if (
            name not in {"stdout.log", "stderr.log", "HOST_SAMPLES.jsonl"}
            or sha256_file(root / SUCCESS / name) != digest
        ):
            raise FourBIntegrityError("E36 successful recorder raw log changed")
    stdout = _load_json(root / SUCCESS / "stdout.log", "successful recorder output")
    raw_profiles = stdout.get("profiles")
    if (
        stdout.get("status") != "BOUNDED_C0_PROFILES_COMPLETE"
        or stdout.get("final_runtime_qualification_published") is not False
        or not isinstance(raw_profiles, list)
        or len(raw_profiles) != len(ladder)
    ):
        raise FourBIntegrityError("E36 successful recorder output does not prove four profiles")
    for step, raw_profile in zip(ladder, raw_profiles, strict=True):
        profile = dict(_mapping(raw_profile, "recorded profile"))
        name = f"{PROFILE_ROOT}/{step}.json"
        if profile.pop("published_path", None) != name or profile != dict(
            _load_json(root / name, "published profile")
        ):
            raise FourBIntegrityError("E36 published profile differs from recorded execution")
    return results


def build_protocol_block_disposition(root: Path) -> dict[str, Any]:
    root = root.resolve()
    registry = load_registry(root / "configs/EXPERIMENT_REGISTRY.yaml")
    entries = [entry for entry in registry.experiments if entry.experiment_id == "E36"]
    if len(entries) != 1 or entries[0].priority != "P1" or entries[0].depends_on != ["E25"]:
        raise FourBIntegrityError("E36 protocol-block disposition is restricted to frozen P1 E36")
    config, _ = _config(root)
    preflight = _verify_preflight(root, _binding(root, E36_PREFLIGHT))
    if preflight.get("status") != "READY_FOR_BOUNDED_CUDA_QUALIFICATION":
        raise FourBIntegrityError("E36 protocol-block disposition requires qualified preflight")
    for definition in config["frozen_recipe"]["definitions"].values():
        completion = _safe_path(root, definition["completion_path"])
        if completion.exists() or (completion.parent / "RUN_IDENTITY.json").exists():
            raise FourBIntegrityError(
                "E36 protocol-block disposition conflicts with scientific run"
            )
    evidence = _audit(root)
    profiles = _profiles(root, list(config["authorized_fallback_ladder"]))
    code = {name: _binding(root, name) for name in CODE_PATHS}
    validation = _load_json(root / VALIDATION, "protocol disposition CPU validation")
    if (
        validation.get("status") != "PASSED"
        or validation.get("exit_codes") != [0, 0, 0]
        or validation.get("code_inventory") != code
        or validation.get("cuda_used") is not False
    ):
        raise FourBIntegrityError("E36 protocol disposition CPU qualification is invalid")
    _verify_binding(root, validation.get("log"), "protocol disposition validation log")
    return {
        "schema_version": 2,
        "experiment_id": "E36",
        "protocol_id": E36_PROTOCOL,
        "disposition_policy_id": POLICY_ID,
        "status": STATUS,
        "block_reason": STATUS,
        "priority": "P1",
        "frozen_recipe": _frozen_recipe(),
        "configuration": _binding(root, E36_CONFIG),
        "preflight": _binding(root, E36_PREFLIGHT),
        "blocked_recipes": ["C1", "C2"],
        "completed_scientific_recipes": [],
        "c0_bounded_real_cuda_qualified": True,
        "c0_scientific_reference_executed": False,
        "four_b_transfer_completed": False,
        "four_b_confirmation_available": False,
        "local_compute_limitation_verified": False,
        "authorized_fallback_ladder_exhausted": False,
        "final_runtime_qualified": False,
        "substitute_experiment_run": False,
        "scientific_configuration_changed": False,
        "administrative_p1_disposition_complete": True,
        "interpretation": "4B bounded inference succeeded. Frozen C1/C2 transfer remains "
        "unexecuted because the 1.7B trained starting adapter is incompatible and no 4B "
        "initialization rule was frozen. No claim of 4B confirmation or compute infeasibility.",
        "protocol_blocker_evidence": evidence,
        "bounded_c0_profiles": profiles,
        "successful_recorder_terminal": _binding(root, f"{SUCCESS}/TERMINAL.json"),
        "preserved_attempts": {
            path.relative_to(root).as_posix(): _binding(root, path.relative_to(root).as_posix())
            for path in sorted((root / "evidence/training/E36/host_memory_recovery_v1").rglob("*"))
            if path.is_file()
        },
        "code_inventory": code,
        "cpu_validation": _binding(root, VALIDATION),
    }


def publish_protocol_block_disposition(root: Path) -> dict[str, Any]:
    value = build_protocol_block_disposition(root)
    destination = root / E36_DISPOSITION
    if destination.exists() and destination.read_bytes() != canonical_json_bytes(value):
        raise FourBIntegrityError("E36 disposition already exists with different content")
    atomic_write_json(destination, value)
    return value


def verify_protocol_block_disposition(root: Path) -> dict[str, Any]:
    observed = _load_json(root / E36_DISPOSITION, "protocol-block disposition")
    rebuilt = build_protocol_block_disposition(root)
    if dict(observed) != rebuilt:
        raise FourBIntegrityError("E36 protocol-block disposition changed or is not canonical")
    return rebuilt
