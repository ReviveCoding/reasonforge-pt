"""CUDA execution helpers for checkpoint/recovery qualification.

The public planner remains CPU-only. Heavy imports occur only in worker functions
that are launched as distinct processes by ``scripts/qualify_checkpoint_resume.py``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

from reasonforge.config import file_sha256, identity_hash
from reasonforge.models import QLoRAModelConfig
from reasonforge.training import (
    NativeCheckpointIdentityCallback,
    SFTRunConfig,
    TelemetryCallback,
    TrainingRunConfig,
    construct_sft_trainer,
    train_with_native_resume,
    validate_native_trainer_checkpoint,
)

from .qualification import QualificationConfigError, load_qualification_suite


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _adapter_inventory(path: Path) -> tuple[str, dict[str, str]]:
    inventory = {
        item.relative_to(path).as_posix(): file_sha256(item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }
    if not inventory:
        raise RuntimeError("qualification final adapter is empty")
    return identity_hash(inventory), inventory


def _final_loss(log_history: list[dict[str, Any]]) -> float:
    losses = [float(item["loss"]) for item in log_history if "loss" in item]
    if not losses:
        raise RuntimeError("qualification Trainer emitted no step loss")
    return losses[-1]


def _trainable_parameter_identity(model: Any) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    elements = 0
    for name, parameter in sorted(model.named_parameters()):
        if not parameter.requires_grad:
            continue
        value = parameter.detach().float().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
        count += 1
        elements += value.numel()
    if count == 0:
        raise RuntimeError("qualification model has no trainable adapter parameters")
    return {
        "sha256": digest.hexdigest(),
        "trainable_tensor_count": count,
        "trainable_element_count": elements,
    }


def trainable_parameter_dtype_inventory(model: Any) -> dict[str, Any]:
    """Bind trainable parameter names/shapes to their runtime dtypes."""

    tensors: dict[str, dict[str, Any]] = {}
    counts: dict[str, int] = {}
    for name, parameter in sorted(model.named_parameters()):
        if not parameter.requires_grad:
            continue
        dtype = str(parameter.dtype)
        tensors[name] = {"dtype": dtype, "shape": list(parameter.shape)}
        counts[dtype] = counts.get(dtype, 0) + 1
    if not tensors:
        raise RuntimeError("qualification model has no trainable parameters for dtype audit")
    return {
        "identity_hash": identity_hash(tensors),
        "tensor_count_by_dtype": counts,
        "trainable_tensor_count": len(tensors),
        "tensors": tensors,
    }


def execute_e27_worker(
    workspace: Path,
    config_path: Path,
    *,
    arm: Literal["reference", "recovery"],
    stage: Literal["full", "controlled-stop", "resume"],
) -> dict[str, Any]:
    """Execute one E27 SFT stage; caller guarantees a separate process per stage."""

    # Reuse the qualified production constructors and canonical TRAIN conversion.
    from transformers import AutoTokenizer, TrainerCallback
    from transformers.trainer_utils import enable_full_determinism
    from trl import SFTTrainer

    from reasonforge.orchestration.posttrain import (
        _assert_cuda_qualified,
        _load_adapter_selection,
        _prepare_dataset,
        load_posttrain_spec,
    )

    root = workspace.resolve()
    suite = load_qualification_suite(config_path)
    spec = suite.experiments["E27"]
    if (arm, stage) not in {
        ("reference", "full"),
        ("recovery", "controlled-stop"),
        ("recovery", "resume"),
    }:
        raise QualificationConfigError(f"invalid E27 arm/stage combination: {arm}/{stage}")
    # Establish RNG and deterministic CUDA controls before any model or adapter
    # construction. v3 proved that identical checkpoint state alone was not
    # enough to distinguish recovery correctness from CUDA kernel nondeterminism.
    enable_full_determinism(spec.seed)
    _assert_cuda_qualified(root)
    source_config = root / spec.source_config
    source = load_posttrain_spec(source_config)
    if source.experiment_id != "E10" or source.adapter_selection_artifact is None:
        raise QualificationConfigError("E27 source is not the qualified E10 SFT recipe")
    if file_sha256(source_config) != spec.source_config_sha256:
        raise QualificationConfigError("E27 source config changed after protocol freeze")
    adapter = _load_adapter_selection(root / source.adapter_selection_artifact, root)
    run_id = spec.reference_run_id if arm == "reference" else spec.recovery_run_id
    output_dir = root / suite.output_root / "E27" / run_id
    if stage != "resume" and output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"E27 stage refuses to overwrite existing output: {output_dir}")
    if stage == "resume" and (output_dir / "RUN_COMPLETE.json").exists():
        raise RuntimeError("E27 recovered run is already complete")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_identity = {
        "experiment_id": "E27",
        "run_id": run_id,
        "protocol_id": suite.protocol_id,
        "suite_config_hash": file_sha256(config_path),
        "source_experiment_id": "E10",
        "source_config_hash": spec.source_config_sha256,
        "model_family": f"{source.model.model_id}@{source.model.revision}",
        "adapter_architecture": identity_hash(adapter.model_dump(mode="json")),
        "objective": "SFT_NATIVE_RESUME_QUALIFICATION",
        "dataset_fingerprint": source.dataset.sha256,
        "data_role": "TRAIN",
        "max_steps": spec.max_steps,
        "save_steps": spec.save_steps,
        "seed": spec.seed,
        "full_determinism": True,
        "resume_adapter_dtype_policy": spec.resume_adapter_dtype_policy,
        "protected_access_state": "NOT_ACCESSED",
    }
    identity_path = output_dir / "RUN_IDENTITY.json"
    if identity_path.is_file():
        existing = json.loads(identity_path.read_text("utf-8"))
        if existing != run_identity:
            raise RuntimeError("E27 resume identity changed")
    else:
        _atomic_json(identity_path, run_identity)
    checkpoint = output_dir / f"checkpoint-{spec.interruption_after_step}"
    native_resume = checkpoint if stage == "resume" else None
    if native_resume is not None:
        validate_native_trainer_checkpoint(native_resume, run_identity)
    training_run = TrainingRunConfig(
        experiment_id="E27",
        run_id=run_id,
        output_dir=output_dir,
        model=QLoRAModelConfig(source.model.model_id, source.model.revision),
        adapter=adapter.runtime_config(),
        dataset_fingerprint=source.dataset.sha256,
        max_steps=spec.max_steps,
        median_optimizer_step_seconds=source.estimated_step_seconds,
        per_device_train_batch_size=source.microbatch,
        gradient_accumulation_steps=source.gradient_accumulation,
        learning_rate=source.learning_rate,
        warmup_steps=min(source.warmup_steps, spec.interruption_after_step),
        seed=spec.seed,
        data_seed=spec.seed,
        native_resume_from_checkpoint=native_resume,
    )

    class ControlledStopCallback(TrainerCallback):  # type: ignore[misc]
        def on_step_end(self, _args: Any, state: Any, control: Any, **_kwargs: Any) -> Any:
            if (
                stage == "controlled-stop"
                and int(state.global_step) >= spec.interruption_after_step
            ):
                control.should_save = True
                control.should_training_stop = True
            return control

    class AuditedSFTTrainer(SFTTrainer):  # type: ignore[misc]
        _rf_batch_trace: list[str]
        _rf_resume_dtype_audit: dict[str, Any] | None = None

        def _load_from_checkpoint(self, resume_from_checkpoint: str, model: Any = None) -> None:
            """Run native loading, then undo PEFT's resume-only adapter upcast.

            Transformers 5.16.1 calls ``PeftModel.load_adapter`` without an
            ``autocast_adapter_dtype`` argument. PEFT 0.20.0 therefore uses its
            true default and converts BF16 adapters to FP32. Fresh SFT
            construction retains BF16 adapters, so that default changes the
            optimizer/model identity only in the recovered process. Restoring
            each trainable tensor to its pre-load dtype here occurs before
            Trainer constructs and restores the optimizer.
            """

            active_model = self.model if model is None else model
            before = trainable_parameter_dtype_inventory(active_model)
            expected = {
                name: parameter.dtype
                for name, parameter in active_model.named_parameters()
                if parameter.requires_grad
            }
            if self.optimizer is not None:
                raise RuntimeError("E27 dtype correction must precede optimizer construction")
            super()._load_from_checkpoint(resume_from_checkpoint, model)
            native_loaded = trainable_parameter_dtype_inventory(active_model)
            observed_names = {
                name
                for name, parameter in active_model.named_parameters()
                if parameter.requires_grad
            }
            if observed_names != set(expected):
                raise RuntimeError("E27 native load changed the trainable parameter set")
            restored = 0
            for name, parameter in active_model.named_parameters():
                if not parameter.requires_grad:
                    continue
                expected_dtype = expected[name]
                if parameter.dtype != expected_dtype:
                    parameter.data = parameter.data.to(dtype=expected_dtype)
                    restored += 1
            corrected = trainable_parameter_dtype_inventory(active_model)
            if corrected != before:
                raise RuntimeError("E27 failed to restore the pre-load adapter dtype identity")
            if restored == 0:
                raise RuntimeError("E27 v6 did not exercise the frozen PEFT dtype correction")
            self._rf_resume_dtype_audit = {
                "status": "CORRECTED_BEFORE_OPTIMIZER_CONSTRUCTION",
                "process_id": os.getpid(),
                "policy": spec.resume_adapter_dtype_policy,
                "native_resume_from": str(resume_from_checkpoint),
                "optimizer_existed_during_correction": False,
                "before_native_load": before,
                "after_native_load_before_correction": native_loaded,
                "after_correction": corrected,
                "restored_tensor_count": restored,
                "native_load_changed_dtype": native_loaded != before,
            }
            dtype_audit_path = output_dir / "RESUME_DTYPE_AUDIT.json"
            _atomic_json(dtype_audit_path, self._rf_resume_dtype_audit)

        def training_step(
            self,
            model: Any,
            inputs: dict[str, Any],
            num_items_in_batch: Any = None,
        ) -> Any:
            digest = hashlib.sha256()
            for name in ("input_ids", "attention_mask", "labels"):
                value = inputs.get(name)
                if value is None:
                    continue
                tensor = value.detach().cpu().contiguous()
                digest.update(name.encode("ascii"))
                digest.update(str(tuple(tensor.shape)).encode("ascii"))
                digest.update(tensor.numpy().tobytes())
            self._rf_batch_trace.append(digest.hexdigest())
            return super().training_step(model, inputs, num_items_in_batch)

    dataset = _prepare_dataset(root / source.dataset.path, source.objective)
    tokenizer = AutoTokenizer.from_pretrained(
        source.model.model_id,
        revision=source.model.revision,
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    callbacks = [
        NativeCheckpointIdentityCallback(output_dir, run_identity),
        ControlledStopCallback(),
        TelemetryCallback(output_dir / f"telemetry_{stage}.jsonl"),
    ]
    trainer = construct_sft_trainer(
        SFTRunConfig(
            training_run,
            max_length=source.max_length,
            checkpoint_save_steps=spec.save_steps,
            checkpoint_save_total_limit=spec.save_total_limit,
        ),
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
        trainer_cls=AuditedSFTTrainer,
    )
    trainer._rf_batch_trace = []
    process_initial = _trainable_parameter_identity(trainer.model)
    initial_path = output_dir / "INITIAL_STATE.json"
    if stage == "resume":
        initial = json.loads(initial_path.read_text("utf-8"))
        if process_initial != initial:
            raise RuntimeError("E27 resume process did not reconstruct the frozen step-0 adapter")
        _atomic_json(output_dir / "RESUME_PROCESS_INITIAL_STATE.json", process_initial)
    else:
        _atomic_json(initial_path, process_initial)
        initial = process_initial
    result = train_with_native_resume(trainer, native_resume)
    batch_trace_path = output_dir / f"BATCH_TRACE_{stage}.json"
    _atomic_json(
        batch_trace_path,
        {
            "stage": stage,
            "process_id": os.getpid(),
            "microbatch_count": len(trainer._rf_batch_trace),
            "batch_hashes": trainer._rf_batch_trace,
        },
    )
    global_step = int(trainer.state.global_step)
    stage_result = {
        "experiment_id": "E27",
        "run_id": run_id,
        "arm": arm,
        "stage": stage,
        "process_id": os.getpid(),
        "global_step": global_step,
        "train_result": dict(getattr(result, "metrics", result)),
        "batch_trace": batch_trace_path.relative_to(root).as_posix(),
        "batch_trace_sha256": file_sha256(batch_trace_path),
    }
    if stage == "controlled-stop":
        if global_step != spec.interruption_after_step:
            raise RuntimeError("controlled E27 stop did not occur at the frozen step")
        validate_native_trainer_checkpoint(checkpoint, run_identity)
        stage_result["status"] = "CONTROLLED_STOP"
        stage_result["checkpoint"] = checkpoint.relative_to(root).as_posix()
        _atomic_json(output_dir / "CONTROLLED_STOP.json", stage_result)
        return stage_result
    if global_step != spec.max_steps:
        raise RuntimeError("E27 final stage did not reach the frozen final step")
    final_adapter = output_dir / "final_adapter"
    trainer.save_model(str(final_adapter))
    tokenizer.save_pretrained(final_adapter)
    adapter_hash, inventory = _adapter_inventory(final_adapter)
    log_history = list(trainer.state.log_history)
    complete = {
        "status": "COMPLETE",
        "experiment_id": "E27",
        "run_id": run_id,
        "arm": arm,
        "stage": stage,
        "process_id": os.getpid(),
        "process_ids": [os.getpid()],
        "protocol_id": suite.protocol_id,
        "suite_config_hash": file_sha256(config_path),
        "final_global_step": global_step,
        "consumed_examples": (global_step * source.microbatch * source.gradient_accumulation),
        "final_loss": _final_loss(log_history),
        "log_history": log_history,
        "final_adapter_path": final_adapter.relative_to(root).as_posix(),
        "parameter_inventory_hash": adapter_hash,
        "parameter_file_inventory": inventory,
        "initial_parameter_identity": initial,
        "resume_process_initial_parameter_identity": process_initial,
        "native_resume_from": (
            native_resume.relative_to(root).as_posix() if native_resume is not None else None
        ),
        "batch_trace": batch_trace_path.relative_to(root).as_posix(),
        "batch_trace_sha256": file_sha256(batch_trace_path),
        "full_determinism": True,
        "final_parameter_dtypes": trainable_parameter_dtype_inventory(trainer.model),
        "resume_dtype_audit": (
            (output_dir / "RESUME_DTYPE_AUDIT.json").relative_to(root).as_posix()
            if trainer._rf_resume_dtype_audit is not None
            else None
        ),
        "resume_dtype_audit_sha256": (
            file_sha256(output_dir / "RESUME_DTYPE_AUDIT.json")
            if trainer._rf_resume_dtype_audit is not None
            else None
        ),
    }
    _atomic_json(output_dir / "RUN_COMPLETE.json", complete)
    return complete


def compare_adapter_parameters(workspace: Path, left: Path, right: Path) -> float:
    """Compute the exact maximum absolute tensor difference for two PEFT adapters."""

    from safetensors.torch import load_file

    left_path = workspace / left / "adapter_model.safetensors"
    right_path = workspace / right / "adapter_model.safetensors"
    left_state = load_file(str(left_path), device="cpu")
    right_state = load_file(str(right_path), device="cpu")
    if set(left_state) != set(right_state):
        raise RuntimeError("qualification adapter tensor keys differ")
    maximum = 0.0
    for name in sorted(left_state):
        if left_state[name].shape != right_state[name].shape:
            raise RuntimeError(f"qualification adapter tensor shape differs: {name}")
        observed = float((left_state[name].float() - right_state[name].float()).abs().max())
        maximum = max(maximum, observed)
    return maximum


def record_e28_partial_interruption(
    workspace: Path,
    config_path: Path,
    *,
    terminate_process: bool = True,
) -> dict[str, Any]:
    """Durably begin a post-checkpoint group, then emulate process loss.

    The caller runs this in its own process. With ``terminate_process=True`` the
    process exits immediately after the atomic partial marker is durable.
    """

    from reasonforge.orchestration.rl import CompletedStepJournal

    root = workspace.resolve()
    suite = load_qualification_suite(config_path)
    spec = suite.experiments["E28"]
    run_dir = root / suite.output_root / "E28" / spec.recovery_run_id
    identity_path = run_dir / "RUN_IDENTITY.json"
    controlled_path = run_dir / "CONTROLLED_STOP.json"
    if not identity_path.is_file() or not controlled_path.is_file():
        raise RuntimeError("E28 partial probe requires the verified controlled-stop stage")
    identity = json.loads(identity_path.read_text("utf-8"))
    controlled = json.loads(controlled_path.read_text("utf-8"))
    if controlled.get("global_step") != spec.interruption_after_step:
        raise RuntimeError("E28 controlled-stop step does not match the frozen protocol")
    journal = CompletedStepJournal(
        root / "artifacts/rl_orchestration",
        experiment_id="E28",
        run_id=spec.recovery_run_id,
        config_hash=identity_hash(identity),
        groups_per_step=int(spec.groups_per_step or 0),
        generations_per_group=int(spec.generations_per_group or 0),
    )
    resume = journal.resume_plan(mutate=True)
    if (
        resume.get("action") != "RESUME"
        or resume.get("completed_optimizer_step") != spec.interruption_after_step
    ):
        raise RuntimeError(f"E28 partial probe found no compatible completed step: {resume}")
    schedule = json.loads((run_dir / "ROLLOUT_SCHEDULE.json").read_text("utf-8"))
    ordinal = journal.completed_groups
    group_ids = schedule.get("group_ids")
    task_ids = schedule.get("task_ids")
    if not isinstance(group_ids, list) or not isinstance(task_ids, list):
        raise RuntimeError("E28 frozen rollout schedule is invalid")
    group_id, task_id = str(group_ids[ordinal]), str(task_ids[ordinal])
    journal.begin_group(
        group_id,
        task_id=task_id,
        artifact_paths=[(run_dir / "ROLLOUT_REWARD_EVIDENCE.jsonl").relative_to(root).as_posix()],
    )
    event = {
        "event": "PARTIAL_ROLLOUT_INTERRUPTED",
        "experiment_id": "E28",
        "run_id": spec.recovery_run_id,
        "process_id": os.getpid(),
        "exit_code": spec.controlled_partial_exit_code,
        "interrupted_group_id": group_id,
        "resume_from_completed_step": spec.interruption_after_step,
        "excluded_from_gradient_accounting": True,
        "partial_marker": journal.partial_path.relative_to(root).as_posix(),
    }
    _atomic_json(run_dir / "PARTIAL_INTERRUPTION.json", event)
    if terminate_process:
        assert spec.controlled_partial_exit_code is not None
        os._exit(spec.controlled_partial_exit_code)
    return event
