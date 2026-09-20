"""Atomic publication and verification of project milestone checkpoints."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from .hashing import atomic_write_bytes, atomic_write_json, hash_inventory, sha256_file
from .models import CheckpointIntegrityError, ProjectCheckpointManifest, utc_now

REQUIRED_PROJECT_FILES = frozenset(
    {
        "CHECKPOINT_MANIFEST.json",
        "STATUS.md",
        "NEXT_ACTION.md",
        "RESUME_INSTRUCTIONS.md",
        "ARTIFACT_HASHES.json",
        "EXPERIMENT_REGISTRY_SNAPSHOT.yaml",
        "BLOCKERS.json",
        "GIT_STATE.txt",
        "ENVIRONMENT_SNAPSHOT.json",
    }
)
MILESTONE_ORDER = {
    "CP00_BOOTSTRAP": 0,
    "CP10_DATA_FREEZE": 10,
    "CP20_REWARD_ENV_QUALIFIED": 20,
    "CP30_POSTTRAIN_BASELINES": 30,
    "CP40_RL_BASELINES": 40,
    "CP50_CANDIDATE_FREEZE": 50,
    "CP60_PRE_PROTECTED": 60,
    "CP70_PROTECTED_COMPLETE": 70,
    "CP80_REPLICATION_COMPLETE": 80,
    "CP90_LOCAL_COMPLETE": 90,
}


class ProjectCheckpointStore:
    """A directory-backed, append-only milestone checkpoint store."""

    def __init__(self, root: Path | str = Path("evidence/checkpoints")) -> None:
        self.root = Path(root)

    @property
    def latest_path(self) -> Path:
        return self.root / "LATEST.json"

    def publish(
        self,
        manifest: ProjectCheckpointManifest,
        *,
        status: str,
        next_action: str,
        resume_instructions: str,
        experiment_registry_snapshot: str,
        blockers: str | bytes = b"{}\n",
        git_state: str = "unknown\n",
        environment_snapshot: str | bytes = b"{}\n",
        extra_artifacts: Mapping[str, str | bytes] | None = None,
    ) -> Path:
        """Stage, validate, atomically publish, then advance ``LATEST.json``."""

        manifest.validate()
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self.root / manifest.checkpoint_id
        if destination.exists():
            raise FileExistsError(f"checkpoint is immutable and already exists: {destination}")
        if self.latest_path.exists():
            previous, _ = self.verify_latest()
            previous_rank = MILESTONE_ORDER.get(previous.checkpoint_id)
            new_rank = MILESTONE_ORDER.get(manifest.checkpoint_id)
            if previous_rank is not None and new_rank is not None and new_rank <= previous_rank:
                raise CheckpointIntegrityError(
                    "project milestone publication must move forward: "
                    f"{previous.checkpoint_id} -> {manifest.checkpoint_id}"
                )
        staging = self.root / f".{manifest.checkpoint_id}.tmp-{uuid.uuid4().hex}"
        staging.mkdir()
        try:
            artifacts: dict[str, str | bytes] = {
                "STATUS.md": status,
                "NEXT_ACTION.md": next_action,
                "RESUME_INSTRUCTIONS.md": resume_instructions,
                "EXPERIMENT_REGISTRY_SNAPSHOT.yaml": experiment_registry_snapshot,
                "BLOCKERS.json": blockers,
                "GIT_STATE.txt": git_state,
                "ENVIRONMENT_SNAPSHOT.json": environment_snapshot,
            }
            extras = dict(extra_artifacts or {})
            reserved = sorted(set(extras) & REQUIRED_PROJECT_FILES)
            if reserved:
                raise ValueError(f"extra artifacts use reserved names: {reserved}")
            artifacts.update(extras)
            for relative, content in artifacts.items():
                target = self._safe_child(staging, relative)
                atomic_write_bytes(
                    target, content.encode("utf-8") if isinstance(content, str) else content
                )

            inventory = sorted(set(artifacts) | REQUIRED_PROJECT_FILES)
            published_manifest = replace(manifest, artifact_inventory=inventory)
            atomic_write_json(staging / "CHECKPOINT_MANIFEST.json", published_manifest.to_dict())
            hashes = hash_inventory(staging, set(inventory) - {"ARTIFACT_HASHES.json"})
            atomic_write_json(staging / "ARTIFACT_HASHES.json", hashes)

            self.verify(staging, expected_checkpoint_id=manifest.checkpoint_id)
            os.replace(staging, destination)
            self.verify(destination, expected_checkpoint_id=manifest.checkpoint_id)
            pointer = {
                "checkpoint_id": manifest.checkpoint_id,
                "path": manifest.checkpoint_id,
                "manifest_sha256": sha256_file(destination / "CHECKPOINT_MANIFEST.json"),
                "verified_at": utc_now(),
                "schema_version": 1,
            }
            atomic_write_json(self.latest_path, pointer)
            return destination
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging)
            raise

    def verify(
        self, checkpoint: Path | str, *, expected_checkpoint_id: str | None = None
    ) -> ProjectCheckpointManifest:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_dir():
            raise CheckpointIntegrityError(f"checkpoint directory is missing: {checkpoint_path}")
        absent = sorted(
            name for name in REQUIRED_PROJECT_FILES if not (checkpoint_path / name).is_file()
        )
        if absent:
            raise CheckpointIntegrityError(f"checkpoint is missing required files: {absent}")
        try:
            manifest_value = json.loads(
                (checkpoint_path / "CHECKPOINT_MANIFEST.json").read_text("utf-8")
            )
            recorded_hashes = json.loads(
                (checkpoint_path / "ARTIFACT_HASHES.json").read_text("utf-8")
            )
            json.loads((checkpoint_path / "BLOCKERS.json").read_text("utf-8"))
            json.loads((checkpoint_path / "ENVIRONMENT_SNAPSHOT.json").read_text("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise CheckpointIntegrityError(f"checkpoint metadata is unreadable: {error}") from error
        try:
            manifest = ProjectCheckpointManifest.from_dict(manifest_value)
        except (KeyError, TypeError, ValueError) as error:
            raise CheckpointIntegrityError(f"invalid checkpoint manifest: {error}") from error
        if expected_checkpoint_id and manifest.checkpoint_id != expected_checkpoint_id:
            raise CheckpointIntegrityError(
                f"checkpoint ID mismatch: {manifest.checkpoint_id!r} != {expected_checkpoint_id!r}"
            )
        if not isinstance(manifest.artifact_inventory, list) or any(
            not isinstance(name, str) for name in manifest.artifact_inventory
        ):
            raise CheckpointIntegrityError("manifest artifact inventory must be a string list")
        if not isinstance(recorded_hashes, dict) or any(
            not isinstance(name, str) or not isinstance(digest, str)
            for name, digest in recorded_hashes.items()
        ):
            raise CheckpointIntegrityError("artifact hashes must be a string mapping")
        declared = set(manifest.artifact_inventory)
        if not REQUIRED_PROJECT_FILES.issubset(declared):
            missing = sorted(REQUIRED_PROJECT_FILES - declared)
            raise CheckpointIntegrityError(f"manifest omits required artifacts: {missing}")
        expected_hashed = declared - {"ARTIFACT_HASHES.json"}
        if set(recorded_hashes) != expected_hashed:
            raise CheckpointIntegrityError(
                "artifact hash inventory does not match manifest inventory"
            )
        try:
            actual_hashes = hash_inventory(checkpoint_path, expected_hashed)
        except (OSError, ValueError) as error:
            raise CheckpointIntegrityError(f"cannot hash checkpoint artifacts: {error}") from error
        mismatched = sorted(
            name for name in expected_hashed if recorded_hashes[name] != actual_hashes[name]
        )
        if mismatched:
            raise CheckpointIntegrityError(f"artifact hash mismatch: {mismatched}")
        return manifest

    def verify_latest(self) -> tuple[ProjectCheckpointManifest, Path]:
        if not self.latest_path.is_file():
            raise CheckpointIntegrityError("LATEST.json does not exist")
        try:
            pointer = json.loads(self.latest_path.read_text("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise CheckpointIntegrityError(f"LATEST.json is unreadable: {error}") from error
        if not isinstance(pointer, dict):
            raise CheckpointIntegrityError("LATEST.json must contain a JSON object")
        relative_path = pointer.get("path")
        if not isinstance(relative_path, str):
            raise CheckpointIntegrityError("LATEST.json has no valid path")
        try:
            checkpoint_path = self._safe_child(self.root, relative_path)
        except ValueError as error:
            raise CheckpointIntegrityError(f"LATEST.json path is unsafe: {error}") from error
        manifest = self.verify(checkpoint_path, expected_checkpoint_id=pointer.get("checkpoint_id"))
        observed = sha256_file(checkpoint_path / "CHECKPOINT_MANIFEST.json")
        if observed != pointer.get("manifest_sha256"):
            raise CheckpointIntegrityError("LATEST.json manifest hash does not match")
        return manifest, checkpoint_path

    @staticmethod
    def _safe_child(root: Path, relative: str) -> Path:
        candidate = (root / relative).resolve()
        resolved_root = root.resolve()
        if candidate == resolved_root or resolved_root not in candidate.parents:
            raise ValueError(f"path escapes checkpoint root: {relative!r}")
        return candidate


def verify_latest(root: Path | str = Path("evidence/checkpoints")) -> dict[str, object]:
    """CLI-friendly verification adapter returning JSON-serializable evidence."""

    manifest, path = ProjectCheckpointStore(root).verify_latest()
    return {
        "status": "VERIFIED",
        "checkpoint_path": str(path),
        "manifest": manifest.to_dict(),
    }
