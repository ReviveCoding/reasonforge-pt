"""Native Trainer resume, telemetry, and completed-step state hooks."""

from __future__ import annotations

import copy
import json
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol

from reasonforge.config import file_sha256, identity_hash


class _TrainerCallback:
    """Dependency-light implementation of the TrainerCallback event protocol."""

    def __getattr__(self, name: str) -> Callable[..., None]:
        if not name.startswith("on_"):
            raise AttributeError(name)

        def no_op(*_args: object, **_kwargs: object) -> None:
            return None

        return no_op


class TrainerLike(Protocol):
    def train(self, *, resume_from_checkpoint: str | None = None) -> Any: ...


class StatefulSampler(Protocol):
    def state_dict(self) -> dict[str, Any]: ...


def train_with_native_resume(trainer: TrainerLike, checkpoint: Path | None) -> Any:
    """Use the pinned Trainer/TRL native resume API without custom serialization."""
    if checkpoint is not None and not checkpoint.is_dir():
        raise FileNotFoundError(f"native resume checkpoint does not exist: {checkpoint}")
    return trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)


def rcac_runtime_state(sampler: StatefulSampler) -> dict[str, object]:
    """Map an RCAC sampler snapshot to runtime-checkpoint material fields."""
    state = copy.deepcopy(sampler.state_dict())
    config = state.get("config")
    tasks = state.get("tasks")
    if not isinstance(config, Mapping) or not isinstance(tasks, Mapping):
        raise ValueError("RCAC state must contain config and tasks mappings")
    candidate_pool = state.get("candidate_pool")
    if not isinstance(candidate_pool, list) or not candidate_pool:
        raise ValueError("RCAC candidate pool must be a non-empty list")

    p_table = {str(key): value["p"] for key, value in tasks.items()}
    q_table = {str(key): value["q"] for key, value in tasks.items()}
    u_table = {str(key): value["u"] for key, value in tasks.items()}
    selection_counts = {str(key): value["selected"] for key, value in tasks.items()}
    required_config = ("tau_q", "p_target", "sigma", "lambda_u")
    missing = [key for key in required_config if key not in config]
    if missing:
        raise ValueError(f"RCAC config is missing fields: {missing}")
    rng_state = state.get("rng_state_b64")
    if not isinstance(rng_state, str) or not rng_state:
        raise ValueError("RCAC sampler RNG state is missing")
    return {
        "candidate_task_pool_identity": identity_hash(sorted(str(item) for item in candidate_pool)),
        "p_i_table_version": identity_hash(p_table),
        "q_i_table_version": identity_hash(q_table),
        "u_i_table_version": identity_hash(u_table),
        "b_i_balancing_state": copy.deepcopy(state.get("family_counts", {})),
        "tau_q": config["tau_q"],
        "p_target": config["p_target"],
        "sigma": config["sigma"],
        "lambda": config["lambda_u"],
        "family_quotas": copy.deepcopy(state.get("family_quotas", {})),
        "selection_counts": selection_counts,
        "curriculum_phase": state.get("curriculum_phase"),
        "sampler_rng_state": rng_state,
        "rcac_state": state,
    }


class CompletedStepStateHook:
    """Snapshot state only at optimizer-step boundaries.

    A rollout group that was started but not completed prevents publication. Its
    partial trajectories must be preserved separately and explicitly discarded.
    """

    def __init__(self, state_getter: Callable[[], Mapping[str, object]]) -> None:
        self._state_getter = state_getter
        self._latest_step: int | None = None
        self._latest_state: dict[str, object] | None = None
        self._active_rollout_group: str | None = None

    @property
    def latest_completed_step(self) -> int | None:
        return self._latest_step

    def begin_rollout_group(self, group_id: str) -> None:
        if not group_id:
            raise ValueError("rollout group id is required")
        if self._active_rollout_group is not None:
            raise RuntimeError("another rollout group is already active")
        self._active_rollout_group = group_id

    def finish_rollout_group(self, group_id: str) -> None:
        if group_id != self._active_rollout_group:
            raise RuntimeError("rollout group completion does not match the active group")
        self._active_rollout_group = None

    def mark_optimizer_step_complete(self, step: int) -> None:
        if step < 1:
            raise ValueError("completed optimizer step must be positive")
        if self._active_rollout_group is not None:
            raise RuntimeError("cannot complete optimizer step during an active rollout group")
        if self._latest_step is not None and step <= self._latest_step:
            raise ValueError("completed optimizer steps must increase monotonically")
        snapshot = copy.deepcopy(dict(self._state_getter()))
        snapshot["optimizer_step_complete"] = True
        snapshot["completed_optimizer_step"] = step
        self._latest_state = snapshot
        self._latest_step = step

    def checkpoint_state(self, step: int) -> dict[str, object]:
        if self._active_rollout_group is not None:
            raise RuntimeError("partial rollout groups are not resumable")
        if self._latest_step != step or self._latest_state is None:
            raise RuntimeError(f"step {step} is not the latest completed optimizer step")
        return copy.deepcopy(self._latest_state)

    def discard_partial_rollout(self) -> dict[str, object]:
        if self._active_rollout_group is None:
            raise RuntimeError("no partial rollout group is active")
        event = {
            "event": "PARTIAL_ROLLOUT_DISCARDED_AFTER_INTERRUPTION",
            "interrupted_group_id": self._active_rollout_group,
            "resume_from_completed_step": self._latest_step or 0,
            "excluded_from_gradient_accounting": True,
            "must_not_double_count": True,
        }
        self._active_rollout_group = None
        return event


class CompletedStepCallback(_TrainerCallback):
    """Bridge Trainer optimizer-step events to a completed-step state hook."""

    def __init__(self, hook: CompletedStepStateHook) -> None:
        self.hook = hook

    def on_step_end(self, _args: Any, state: Any, control: Any, **_kwargs: Any) -> Any:
        self.hook.mark_optimizer_step_complete(int(state.global_step))
        return control


@dataclass(frozen=True, slots=True)
class TelemetryEvent:
    event: str
    wall_seconds: float
    global_step: int
    metrics: dict[str, object]


class TelemetryCallback(_TrainerCallback):
    """Append compact trainer and optional GPU telemetry as durable JSONL."""

    def __init__(
        self,
        output: Path,
        *,
        clock: Callable[[], float] = time.perf_counter,
        gpu_probe: Callable[[], Mapping[str, object]] | None = None,
    ) -> None:
        self.output = output
        self.clock = clock
        self.gpu_probe = gpu_probe or self._default_gpu_probe
        self._started_at: float | None = None

    @staticmethod
    def _default_gpu_probe() -> Mapping[str, object]:
        try:
            torch = import_module("torch")
        except ImportError:
            return {"cuda_available": False}
        if not torch.cuda.is_available():
            return {"cuda_available": False}
        return {
            "cuda_available": True,
            "device_index": torch.cuda.current_device(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }

    def _write(self, event: TelemetryEvent) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with self.output.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(asdict(event), sort_keys=True, default=str) + "\n")

    def _event(self, name: str, state: Any, metrics: Mapping[str, object]) -> TelemetryEvent:
        now = self.clock()
        started = self._started_at if self._started_at is not None else now
        combined = dict(metrics)
        combined.update(self.gpu_probe())
        return TelemetryEvent(
            event=name,
            wall_seconds=max(0.0, now - started),
            global_step=int(getattr(state, "global_step", 0)),
            metrics=combined,
        )

    def on_train_begin(self, _args: Any, state: Any, control: Any, **_kwargs: Any) -> Any:
        self._started_at = self.clock()
        self._write(self._event("train_begin", state, {}))
        return control

    def on_log(
        self,
        _args: Any,
        state: Any,
        control: Any,
        logs: Mapping[str, object] | None = None,
        **_kwargs: Any,
    ) -> Any:
        self._write(self._event("log", state, logs or {}))
        return control

    def on_save(self, _args: Any, state: Any, control: Any, **_kwargs: Any) -> Any:
        self._write(self._event("save", state, {}))
        return control

    def on_train_end(self, _args: Any, state: Any, control: Any, **_kwargs: Any) -> Any:
        self._write(self._event("train_end", state, {}))
        return control


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


class NativeCheckpointIdentityCallback(_TrainerCallback):
    """Bind each completed native Trainer checkpoint to its scientific identity."""

    def __init__(self, output_dir: Path, run_identity: Mapping[str, object]) -> None:
        self.output_dir = output_dir
        self.run_identity = copy.deepcopy(dict(run_identity))
        self.identity_hash = identity_hash(self.run_identity)

    def on_save(self, _args: Any, state: Any, control: Any, **_kwargs: Any) -> Any:
        step = int(state.global_step)
        checkpoint = self.output_dir / f"checkpoint-{step}"
        if not checkpoint.is_dir():
            raise RuntimeError(f"Trainer save callback cannot locate {checkpoint}")
        trainer_state = checkpoint / "trainer_state.json"
        if not trainer_state.is_file() or trainer_state.stat().st_size == 0:
            raise RuntimeError("native checkpoint has no complete trainer_state.json")
        _atomic_json(
            checkpoint / "REASONFORGE_CHECKPOINT.json",
            {
                "status": "COMPLETE",
                "checkpoint_step": step,
                "run_identity": self.run_identity,
                "run_identity_hash": self.identity_hash,
            },
        )
        publish_native_checkpoint_completion(checkpoint, self.run_identity)
        return control


def _native_checkpoint_inventory(checkpoint: Path) -> dict[str, dict[str, str | int]]:
    inventory: dict[str, dict[str, str | int]] = {}
    for path in sorted(checkpoint.rglob("*")):
        if (
            not path.is_file()
            or path.name == "CHECKPOINT_COMPLETE.json"
            or path.name.startswith(".")
        ):
            continue
        inventory[path.relative_to(checkpoint).as_posix()] = {
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
    return inventory


def publish_native_checkpoint_completion(
    checkpoint: Path, expected_identity: Mapping[str, object]
) -> dict[str, object]:
    """Verify a native Trainer save and atomically publish its durable completion marker."""

    marker_path = checkpoint / "REASONFORGE_CHECKPOINT.json"
    trainer_state_path = checkpoint / "trainer_state.json"
    if not marker_path.is_file() or not trainer_state_path.is_file():
        raise ValueError("native checkpoint lacks Trainer or ReasonForge metadata")
    marker = json.loads(marker_path.read_text("utf-8"))
    trainer_state = json.loads(trainer_state_path.read_text("utf-8"))
    expected_hash = identity_hash(dict(expected_identity))
    step = marker.get("checkpoint_step")
    if (
        marker.get("status") != "COMPLETE"
        or marker.get("run_identity_hash") != expected_hash
        or not isinstance(step, int)
        or trainer_state.get("global_step") != step
    ):
        raise ValueError("native checkpoint identity or step is inconsistent")
    incomplete = [path.name for path in checkpoint.rglob("*") if path.name.endswith(".incomplete")]
    if incomplete:
        raise ValueError(f"native checkpoint retains incomplete sentinels: {incomplete}")
    inventory = _native_checkpoint_inventory(checkpoint)
    required = {"trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"}
    missing = required - set(inventory)
    model_files = {"adapter_model.safetensors", "model.safetensors", "pytorch_model.bin"}
    if missing or not (model_files & set(inventory)):
        raise ValueError(
            f"native checkpoint files are incomplete: missing={sorted(missing)}, "
            "model_or_adapter_weights_present="
            f"{bool(model_files & set(inventory))}"
        )
    completion: dict[str, object] = {
        "status": "COMPLETE",
        "checkpoint_step": step,
        "run_identity_hash": expected_hash,
        "completion_timestamp": datetime.now(UTC).isoformat(),
        "checkpoint_byte_size": sum(int(item["bytes"]) for item in inventory.values()),
        "file_inventory": inventory,
    }
    _atomic_json(checkpoint / "CHECKPOINT_COMPLETE.json", completion)
    return completion


def validate_native_trainer_checkpoint(
    checkpoint: Path, expected_identity: Mapping[str, object]
) -> int:
    """Validate a completed-step Trainer checkpoint before native resume."""

    if not checkpoint.is_dir():
        raise FileNotFoundError(f"native checkpoint does not exist: {checkpoint}")
    marker_path = checkpoint / "REASONFORGE_CHECKPOINT.json"
    trainer_state_path = checkpoint / "trainer_state.json"
    completion_path = checkpoint / "CHECKPOINT_COMPLETE.json"
    if (
        not marker_path.is_file()
        or not trainer_state_path.is_file()
        or not completion_path.is_file()
    ):
        raise ValueError("native checkpoint is incomplete or lacks ReasonForge identity")
    try:
        marker = json.loads(marker_path.read_text("utf-8"))
        trainer_state = json.loads(trainer_state_path.read_text("utf-8"))
        completion = json.loads(completion_path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"native checkpoint metadata is invalid: {error}") from error
    expected_hash = identity_hash(dict(expected_identity))
    if marker.get("status") != "COMPLETE" or marker.get("run_identity_hash") != expected_hash:
        raise ValueError("native checkpoint scientific identity mismatch")
    step = marker.get("checkpoint_step")
    if not isinstance(step, int) or step < 1 or trainer_state.get("global_step") != step:
        raise ValueError("native checkpoint step is inconsistent")
    try:
        directory_step = int(checkpoint.name.removeprefix("checkpoint-"))
    except ValueError as error:
        raise ValueError("native checkpoint directory has an invalid step suffix") from error
    if directory_step != step:
        raise ValueError("native checkpoint directory and manifest steps differ")
    if completion.get("status") != "COMPLETE" or completion.get("checkpoint_step") != step:
        raise ValueError("native checkpoint completion marker is inconsistent")
    if completion.get("run_identity_hash") != expected_hash:
        raise ValueError("native checkpoint completion identity mismatch")
    inventory = completion.get("file_inventory")
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("native checkpoint completion inventory is missing")
    observed = _native_checkpoint_inventory(checkpoint)
    if inventory != observed:
        raise ValueError("native checkpoint file inventory or hashes changed after completion")
    return step
