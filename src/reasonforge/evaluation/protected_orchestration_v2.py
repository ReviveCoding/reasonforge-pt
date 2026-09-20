"""CP60 recovery-amendment gate and journaled protected generation orchestration."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from reasonforge.checkpoints.hashing import atomic_write_json, sha256_file

from .protected_outputs import (
    ProtectedOutputIntegrityError,
    ProtectedOutputRecord,
    ProtectedOutputVerification,
    ProtectedOutputWriter,
)
from .protected_outputs_v2 import JournaledProtectedOutputWriter
from .protected_protocol import (
    FrozenProtectedRunPlan,
    PredictionFunction,
    ProtectedBenchmarkSpec,
    require_verified_cp60,
    verify_run_plan,
)

RECOVERY_AMENDMENT_ID = "RFPT-PROTECTED-RECOVERY-v2"


def verify_recovery_amendment(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise ProtectedOutputIntegrityError("recovery amendment must be an object")
    if (
        value.get("schema_version") != 1
        or value.get("status") != "VERIFIED_RECOVERY_AMENDMENT"
        or value.get("recovery_protocol_id") != RECOVERY_AMENDMENT_ID
        or value.get("generation_semantics_changed") is not False
        or value.get("regeneration_authorized") is not False
    ):
        raise ProtectedOutputIntegrityError("invalid protected recovery amendment")
    inventory = value.get("recovery_code_inventory")
    if not isinstance(inventory, Mapping):
        raise ProtectedOutputIntegrityError("recovery amendment has no code inventory")
    for raw_path, expected in inventory.items():
        candidate = Path(str(raw_path))
        if not candidate.is_file() or sha256_file(candidate) != expected:
            raise ProtectedOutputIntegrityError(f"recovery code identity mismatch: {candidate}")
    frozen_inventory = value.get("parent_frozen_inventory")
    if not isinstance(frozen_inventory, Mapping):
        raise ProtectedOutputIntegrityError("recovery amendment has no parent frozen inventory")
    for raw_path, expected in frozen_inventory.items():
        candidate = Path(str(raw_path))
        if not candidate.is_file() or sha256_file(candidate) != expected:
            raise ProtectedOutputIntegrityError(f"parent frozen identity mismatch: {candidate}")
    recovery_config = value.get("recovery_config")
    if not isinstance(recovery_config, Mapping):
        raise ProtectedOutputIntegrityError("recovery amendment has no configuration identity")
    config_path = Path(str(recovery_config.get("path")))
    if not config_path.is_file() or sha256_file(config_path) != recovery_config.get("sha256"):
        raise ProtectedOutputIntegrityError("recovery configuration identity mismatch")
    for prefix in ("cp60_manifest", "cp60_bindings"):
        bound = value.get(prefix)
        if not isinstance(bound, Mapping):
            raise ProtectedOutputIntegrityError(f"recovery amendment has no {prefix} identity")
        bound_path = Path(str(bound.get("path")))
        if not bound_path.is_file() or sha256_file(bound_path) != bound.get("sha256"):
            raise ProtectedOutputIntegrityError(f"{prefix} identity mismatch")
    return value


def run_protected_predictions_v2(
    *,
    plan: FrozenProtectedRunPlan,
    spec: ProtectedBenchmarkSpec,
    output_root: Path,
    predict: PredictionFunction,
    recovery_amendment: Path,
    shard_max_records: int = 32,
) -> ProtectedOutputVerification:
    """Resume or generate with unchanged inference semantics and journaled publication."""

    require_verified_cp60()
    verify_recovery_amendment(recovery_amendment)
    examples = verify_run_plan(plan, spec)
    destination = output_root / plan.model_id / plan.output_set_id
    if destination.exists():
        plan_bytes_sha256 = sha256_file(destination / "RUN_PLAN.json")
        if (destination / "WRITER_V2_ADOPTION.json").is_file():
            writer = JournaledProtectedOutputWriter.open(destination)
        else:
            writer = JournaledProtectedOutputWriter.adopt(
                destination, run_plan_sha256=plan_bytes_sha256
            )
        verification = writer.verify()
        if verification.manifest.identity != plan.identity:
            raise ProtectedOutputIntegrityError("existing output identity differs from frozen plan")
    else:
        base = ProtectedOutputWriter.create(
            output_root,
            plan.identity,
            plan.expected_example_ids,
            shard_max_records=shard_max_records,
        )
        atomic_write_json(base.output_directory / "RUN_PLAN.json", plan.to_dict())
        plan_bytes_sha256 = sha256_file(base.output_directory / "RUN_PLAN.json")
        writer = JournaledProtectedOutputWriter.adopt(
            base.output_directory, run_plan_sha256=plan_bytes_sha256
        )
    existing_plan = json.loads((destination / "RUN_PLAN.json").read_text("utf-8"))
    if existing_plan != plan.to_dict():
        raise ProtectedOutputIntegrityError("output RUN_PLAN differs from frozen plan")
    missing = set(writer.verify().missing_example_ids)
    by_id = {item.example_id: item for item in examples}
    for example_id in plan.expected_example_ids:
        if example_id not in missing:
            continue
        output_text, metadata = predict(by_id[example_id], spec.generation_mode)
        writer.append(
            ProtectedOutputRecord(
                output_set_id=plan.output_set_id,
                protocol_id=plan.protocol_id,
                model_id=plan.model_id,
                model_checkpoint_sha256=plan.model_checkpoint_sha256,
                generation_config_id=str(plan.generation_config["generation_config_id"]),
                generation_config_sha256=plan.generation_config_sha256,
                example_id=example_id,
                benchmark_id=spec.benchmark_id,
                benchmark_revision=spec.revision,
                prompt_template_id=spec.prompt_template_id,
                output_text=output_text,
                finish_reason=str(metadata.pop("finish_reason", "stop")),
                metadata={
                    "example_identity_sha256": plan.example_identity_sha256[example_id],
                    "generation_mode": spec.generation_mode,
                    **metadata,
                },
            )
        )
    return writer.complete()
