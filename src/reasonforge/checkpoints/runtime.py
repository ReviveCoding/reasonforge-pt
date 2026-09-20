"""Runtime checkpoint completion, validation, and recovery event semantics."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from .hashing import atomic_write_bytes, atomic_write_json, hash_inventory, sha256_json
from .models import (
    CheckpointIntegrityError,
    ResumeIdentityMismatch,
    RunIdentity,
    RuntimeCheckpointKind,
    RuntimeCheckpointManifest,
    utc_now,
)

RCAC_STATE_FIELDS = frozenset(
    {
        "candidate_task_pool_identity",
        "p_i_table_version",
        "q_i_table_version",
        "u_i_table_version",
        "b_i_balancing_state",
        "tau_q",
        "p_target",
        "sigma",
        "lambda",
        "family_quotas",
        "selection_counts",
        "curriculum_phase",
        "sampler_rng_state",
    }
)
STATEFUL_STATE_FIELDS = frozenset(
    {
        "scenario_id",
        "initial_state_identity",
        "environment_seed",
        "tool_registry_revision",
        "completed_transitions",
        "current_task_state",
        "authorization_state",
        "persistent_state_snapshot",
        "reward_relevant_state",
    }
)


def validate_additional_state(kind: RuntimeCheckpointKind, state: Mapping[str, object]) -> None:
    if not isinstance(state, Mapping):
        raise CheckpointIntegrityError("runtime additional state must be a mapping")
    required: set[str] = set()
    if kind in {RuntimeCheckpointKind.RCAC, RuntimeCheckpointKind.STATEFUL_RCAC}:
        required.update(RCAC_STATE_FIELDS)
    if kind in {RuntimeCheckpointKind.STATEFUL_GRPO, RuntimeCheckpointKind.STATEFUL_RCAC}:
        required.update(STATEFUL_STATE_FIELDS)
    missing = sorted(name for name in required if name not in state or state[name] is None)
    if missing:
        raise CheckpointIntegrityError(
            f"{kind.value} checkpoint is missing scientifically material state: {missing}"
        )
    if state.get("optimizer_step_complete") is not True and kind in {
        RuntimeCheckpointKind.GRPO,
        RuntimeCheckpointKind.RCAC,
        RuntimeCheckpointKind.STATEFUL_GRPO,
        RuntimeCheckpointKind.STATEFUL_RCAC,
    }:
        raise CheckpointIntegrityError(
            "GRPO-family checkpoints must end at a completed optimizer step"
        )


class RuntimeCheckpointStore:
    """Publish immutable runtime checkpoints beneath per-run state directories."""

    def __init__(self, root: Path | str = Path("artifacts/checkpoints")) -> None:
        self.root = Path(root)

    def publish(
        self,
        manifest: RuntimeCheckpointManifest,
        files: Mapping[str, bytes | str],
    ) -> Path:
        staging = self.begin(manifest)
        try:
            collisions = sorted(set(files) & self._reserved_names())
            if collisions:
                raise ValueError(f"payload uses reserved checkpoint names: {collisions}")
            for relative, content in files.items():
                target = self._safe_child(staging, relative)
                atomic_write_bytes(
                    target, content.encode("utf-8") if isinstance(content, str) else content
                )
            return self.complete(manifest, staging)
        except Exception as error:
            if staging.exists() and staging.parent.name == "IN_PROGRESS":
                self.mark_invalid(staging, reason=f"payload publication failed: {error}")
            raise

    def begin(self, manifest: RuntimeCheckpointManifest) -> Path:
        """Create an IN_PROGRESS directory for a native Trainer/TRL save.

        Callers may write large checkpoint files directly into the returned
        directory and then call :meth:`complete`, avoiding an in-memory copy.
        """

        if not isinstance(manifest.additional_state, dict):
            raise CheckpointIntegrityError("runtime additional state must be a mapping")
        validate_additional_state(manifest.kind, manifest.additional_state)
        self._validate_component("experiment_id", manifest.experiment_id)
        self._validate_component("run_id", manifest.run_id)
        if manifest.checkpoint_step < 0:
            raise CheckpointIntegrityError("checkpoint step cannot be negative")
        run_root = self.root / manifest.experiment_id / manifest.run_id
        progress_root = run_root / "IN_PROGRESS"
        complete_root = run_root / "COMPLETE"
        progress_root.mkdir(parents=True, exist_ok=True)
        complete_root.mkdir(parents=True, exist_ok=True)
        name = f"checkpoint_{manifest.checkpoint_step:08d}"
        destination = complete_root / name
        if destination.exists():
            raise FileExistsError(f"runtime checkpoint already exists: {destination}")
        staging = progress_root / f"{name}.tmp-{uuid.uuid4().hex}"
        staging.mkdir()
        return staging

    def complete(self, manifest: RuntimeCheckpointManifest, staging: Path | str) -> Path:
        """Verify and atomically promote a native IN_PROGRESS checkpoint."""

        validate_additional_state(manifest.kind, manifest.additional_state)
        path = Path(staging)
        expected_progress = (
            self.root / manifest.experiment_id / manifest.run_id / "IN_PROGRESS"
        ).resolve()
        if not path.is_dir() or path.resolve().parent != expected_progress:
            raise CheckpointIntegrityError(
                "staging checkpoint is not in the run's IN_PROGRESS directory"
            )
        discovered = sorted(
            item.relative_to(path).as_posix()
            for item in path.rglob("*")
            if item.is_file() and item.name not in self._reserved_names()
        )
        if not discovered:
            error = CheckpointIntegrityError("runtime checkpoint contains no payload files")
            self.mark_invalid(path, reason=str(error))
            raise error
        if manifest.expected_files and set(manifest.expected_files) != set(discovered):
            error = CheckpointIntegrityError(
                "declared runtime payload inventory differs from staged files"
            )
            self.mark_invalid(path, reason=str(error))
            raise error
        name = f"checkpoint_{manifest.checkpoint_step:08d}"
        destination = expected_progress.parent / "COMPLETE" / name
        if destination.exists():
            raise FileExistsError(f"runtime checkpoint already exists: {destination}")
        try:
            payload_hashes = hash_inventory(path, discovered)
            byte_size = sum((path / item).stat().st_size for item in discovered)
            complete_manifest = replace(
                manifest,
                expected_files=discovered,
                checkpoint_byte_size=byte_size,
            )
            atomic_write_json(path / "PAYLOAD_HASHES.json", payload_hashes)
            atomic_write_json(path / "RUNTIME_MANIFEST.json", complete_manifest.to_dict())
            completion = {
                "experiment_id": manifest.experiment_id,
                "run_id": manifest.run_id,
                "checkpoint_step": manifest.checkpoint_step,
                "manifest_hash": sha256_json(complete_manifest.to_dict()),
                "completed_at": utc_now(),
            }
            atomic_write_json(path / "CHECKPOINT_COMPLETE.json", completion)
            self.verify(path, expected_identity=manifest.identity)
            os.replace(path, destination)
            self.verify(destination, expected_identity=manifest.identity)
            return destination
        except Exception as error:
            if path.exists():
                self.mark_invalid(path, reason=f"checkpoint verification failed: {error}")
            raise

    def mark_invalid(self, staging: Path | str, *, reason: str) -> Path:
        """Preserve an incomplete/broken checkpoint as explicit failure evidence."""

        path = Path(staging)
        if not path.is_dir() or path.parent.name != "IN_PROGRESS":
            raise CheckpointIntegrityError("only an IN_PROGRESS checkpoint may be marked invalid")
        invalid_root = path.parent.parent / "INVALID"
        invalid_root.mkdir(parents=True, exist_ok=True)
        invalid = invalid_root / f"{path.name}-{uuid.uuid4().hex[:8]}"
        os.replace(path, invalid)
        atomic_write_json(
            invalid / "CHECKPOINT_INVALID.json",
            {"status": "CHECKPOINT_INVALID", "reason": reason, "recorded_at": utc_now()},
        )
        return invalid

    def verify(
        self, checkpoint: Path | str, *, expected_identity: RunIdentity | None = None
    ) -> RuntimeCheckpointManifest:
        path = Path(checkpoint)
        required = ["RUNTIME_MANIFEST.json", "PAYLOAD_HASHES.json", "CHECKPOINT_COMPLETE.json"]
        absent = [name for name in required if not (path / name).is_file()]
        if absent:
            raise CheckpointIntegrityError(f"incomplete runtime checkpoint; missing {absent}")
        if (path / "CHECKPOINT_INVALID.json").exists():
            raise CheckpointIntegrityError("runtime checkpoint is marked invalid")
        try:
            manifest = RuntimeCheckpointManifest.from_dict(
                json.loads((path / "RUNTIME_MANIFEST.json").read_text("utf-8"))
            )
            hashes = json.loads((path / "PAYLOAD_HASHES.json").read_text("utf-8"))
            completion = json.loads((path / "CHECKPOINT_COMPLETE.json").read_text("utf-8"))
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            raise CheckpointIntegrityError(
                f"invalid runtime checkpoint metadata: {error}"
            ) from error
        validate_additional_state(manifest.kind, manifest.additional_state)
        if not isinstance(hashes, dict) or any(
            not isinstance(name, str) or not isinstance(digest, str)
            for name, digest in hashes.items()
        ):
            raise CheckpointIntegrityError("runtime payload hashes must be a string mapping")
        if not isinstance(completion, dict):
            raise CheckpointIntegrityError("runtime completion sentinel must be a JSON object")
        if manifest.dataset_fingerprint != manifest.identity.dataset_fingerprint:
            raise CheckpointIntegrityError("manifest and run identity dataset fingerprints differ")
        required_text = {
            "experiment_id": manifest.experiment_id,
            "run_id": manifest.run_id,
            "protocol_id": manifest.protocol_id,
            "training_config_hash": manifest.training_config_hash,
            "source_revision": manifest.source_revision,
            "starting_checkpoint_hash": manifest.starting_checkpoint_hash,
            "sampler_state_hash": manifest.sampler_state_hash,
            "reward_config_hash": manifest.reward_config_hash,
            "hardware_environment_identity": manifest.hardware_environment_identity,
        }
        empty = sorted(name for name, value in required_text.items() if not value)
        if empty:
            raise CheckpointIntegrityError(f"runtime manifest has empty required fields: {empty}")
        if expected_identity is not None:
            expected_identity.assert_compatible(manifest.identity)
        if manifest.checkpoint_step < 0:
            raise CheckpointIntegrityError("checkpoint step cannot be negative")
        if not isinstance(manifest.expected_files, list) or any(
            not isinstance(name, str) or not name for name in manifest.expected_files
        ):
            raise CheckpointIntegrityError("runtime expected files must be a non-empty string list")
        if not manifest.expected_files:
            raise CheckpointIntegrityError("runtime checkpoint has no declared payload files")
        if set(hashes) != set(manifest.expected_files):
            raise CheckpointIntegrityError("runtime payload inventory mismatch")
        try:
            actual = hash_inventory(path, manifest.expected_files)
        except (OSError, ValueError) as error:
            raise CheckpointIntegrityError(f"cannot hash runtime payload: {error}") from error
        mismatch = sorted(name for name in actual if actual[name] != hashes[name])
        if mismatch:
            raise CheckpointIntegrityError(f"runtime payload hash mismatch: {mismatch}")
        if any((path / name).stat().st_size == 0 for name in manifest.expected_files):
            raise CheckpointIntegrityError("runtime checkpoint contains an empty payload file")
        actual_size = sum((path / name).stat().st_size for name in manifest.expected_files)
        if actual_size != manifest.checkpoint_byte_size:
            raise CheckpointIntegrityError("runtime checkpoint byte size mismatch")
        if completion.get("manifest_hash") != sha256_json(manifest.to_dict()):
            raise CheckpointIntegrityError("runtime completion sentinel does not bind manifest")
        for key in ("experiment_id", "run_id", "checkpoint_step"):
            if completion.get(key) != getattr(manifest, key):
                raise CheckpointIntegrityError(f"runtime completion sentinel mismatch: {key}")
        return manifest

    def latest_compatible(
        self, experiment_id: str, run_id: str, identity: RunIdentity
    ) -> tuple[Path, RuntimeCheckpointManifest] | None:
        complete = self.root / experiment_id / run_id / "COMPLETE"
        if not complete.is_dir():
            return None
        candidates: list[tuple[int, Path, RuntimeCheckpointManifest]] = []
        for path in complete.glob("checkpoint_*"):
            if not path.is_dir():
                continue
            try:
                manifest = self.verify(path, expected_identity=identity)
            except (CheckpointIntegrityError, ResumeIdentityMismatch, ValueError, TypeError):
                # A bad or incompatible candidate must not prevent selection of an
                # older compatible COMPLETE checkpoint.
                continue
            candidates.append((manifest.checkpoint_step, path, manifest))
        if not candidates:
            return None
        _, path, manifest = max(candidates, key=lambda value: value[0])
        return path, manifest

    def record_partial_rollout_discard(
        self,
        experiment_id: str,
        run_id: str,
        *,
        interrupted_group_id: str,
        resume_from_completed_step: int,
        partial_trajectory_paths: list[str] | None = None,
    ) -> Path:
        """Record that an incomplete group is preserved but excluded from gradients."""

        if resume_from_completed_step < 0:
            raise ValueError("resume_from_completed_step cannot be negative")
        events = self.root / experiment_id / run_id / "events"
        event_id = uuid.uuid4().hex
        path = events / f"partial_rollout_discarded_{event_id}.json"
        atomic_write_json(
            path,
            {
                "event": "PARTIAL_ROLLOUT_DISCARDED_AFTER_INTERRUPTION",
                "interrupted_group_id": interrupted_group_id,
                "resume_from_completed_step": resume_from_completed_step,
                "partial_trajectory_paths": partial_trajectory_paths or [],
                "excluded_from_gradient_accounting": True,
                "must_not_double_count": True,
                "recorded_at": utc_now(),
            },
        )
        return path

    @staticmethod
    def _safe_child(root: Path, relative: str) -> Path:
        candidate = (root / relative).resolve()
        resolved_root = root.resolve()
        if candidate == resolved_root or resolved_root not in candidate.parents:
            raise ValueError(f"path escapes runtime checkpoint root: {relative!r}")
        return candidate

    @staticmethod
    def _reserved_names() -> frozenset[str]:
        return frozenset(
            {
                "RUNTIME_MANIFEST.json",
                "PAYLOAD_HASHES.json",
                "CHECKPOINT_COMPLETE.json",
                "CHECKPOINT_INVALID.json",
            }
        )

    @staticmethod
    def _validate_component(name: str, value: str) -> None:
        if not value or value in {".", ".."} or "/" in value or "\\" in value:
            raise CheckpointIntegrityError(f"{name} must be a non-empty path component")
