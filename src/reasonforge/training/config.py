"""Frozen run configurations and timing-derived checkpoint cadence."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from reasonforge.config import identity_hash
from reasonforge.models import LoRAAdapterConfig, QLoRAModelConfig


@dataclass(frozen=True, slots=True)
class SaveInterval:
    save_steps: int
    save_total_limit: int
    estimated_checkpoints: int
    estimated_total_seconds: float
    target_interval_seconds: float


def derive_save_interval(
    *,
    median_optimizer_step_seconds: float,
    total_optimizer_steps: int,
    median_save_seconds: float = 0.0,
    target_long_run_minutes: float = 25.0,
    max_checkpoints: int = 10,
    save_total_limit: int = 4,
) -> SaveInterval:
    """Translate a timing pilot into a bounded native ``save_steps`` policy."""
    if median_optimizer_step_seconds <= 0.0:
        raise ValueError("median_optimizer_step_seconds must be positive")
    if total_optimizer_steps < 1:
        raise ValueError("total_optimizer_steps must be positive")
    if median_save_seconds < 0.0:
        raise ValueError("median_save_seconds cannot be negative")
    if target_long_run_minutes <= 0.0:
        raise ValueError("target_long_run_minutes must be positive")
    if max_checkpoints < 1 or save_total_limit < 1:
        raise ValueError("checkpoint and retention limits must be positive")

    total_seconds = median_optimizer_step_seconds * total_optimizer_steps
    if total_seconds <= 20.0 * 60.0:
        target_seconds = total_seconds
        save_steps = total_optimizer_steps
    elif total_seconds <= 60.0 * 60.0:
        target_seconds = total_seconds / 2.0
        save_steps = math.ceil(total_optimizer_steps / 2)
    else:
        target_seconds = target_long_run_minutes * 60.0
        save_steps = max(1, round(target_seconds / median_optimizer_step_seconds))
        if math.ceil(total_optimizer_steps / save_steps) > max_checkpoints:
            save_steps = math.ceil(total_optimizer_steps / max_checkpoints)

    # Keep checkpoint I/O below roughly 10% of the interval when timing exists.
    if median_save_seconds:
        save_steps = max(
            save_steps,
            math.ceil((10.0 * median_save_seconds) / median_optimizer_step_seconds),
        )
    save_steps = min(save_steps, total_optimizer_steps)
    return SaveInterval(
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        estimated_checkpoints=math.ceil(total_optimizer_steps / save_steps),
        estimated_total_seconds=total_seconds,
        target_interval_seconds=target_seconds,
    )


@dataclass(frozen=True, slots=True)
class TrainingRunConfig:
    experiment_id: str
    run_id: str
    output_dir: Path
    model: QLoRAModelConfig
    adapter: LoRAAdapterConfig
    dataset_fingerprint: str
    max_steps: int
    median_optimizer_step_seconds: float
    median_save_seconds: float = 0.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-5
    warmup_steps: int = 0
    weight_decay: float = 0.0
    logging_steps: int = 1
    seed: int = 42
    data_seed: int = 42
    gradient_checkpointing: bool = True
    native_resume_from_checkpoint: Path | None = None

    def __post_init__(self) -> None:
        required = {
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "dataset_fingerprint": self.dataset_fingerprint,
        }
        empty = sorted(name for name, value in required.items() if not value.strip())
        if empty:
            raise ValueError(f"required run identities are empty: {empty}")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.per_device_train_batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("microbatch and gradient accumulation must be positive")
        if self.learning_rate <= 0.0 or self.warmup_steps < 0 or self.weight_decay < 0.0:
            raise ValueError("invalid optimizer configuration")
        if self.logging_steps < 1:
            raise ValueError("logging_steps must be positive")
        if (
            self.native_resume_from_checkpoint is not None
            and not self.native_resume_from_checkpoint.is_dir()
        ):
            raise ValueError("native resume checkpoint must be an existing directory")
        derive_save_interval(
            median_optimizer_step_seconds=self.median_optimizer_step_seconds,
            total_optimizer_steps=self.max_steps,
            median_save_seconds=self.median_save_seconds,
        )

    @property
    def global_batch_size(self) -> int:
        """Single-process global batch; multi-GPU scaling is intentionally absent."""
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    @property
    def save_interval(self) -> SaveInterval:
        return derive_save_interval(
            median_optimizer_step_seconds=self.median_optimizer_step_seconds,
            total_optimizer_steps=self.max_steps,
            median_save_seconds=self.median_save_seconds,
        )

    @property
    def config_hash(self) -> str:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["native_resume_from_checkpoint"] = (
            str(self.native_resume_from_checkpoint)
            if self.native_resume_from_checkpoint is not None
            else None
        )
        return identity_hash(payload)

    def common_trainer_kwargs(self) -> dict[str, Any]:
        cadence = self.save_interval
        return {
            "output_dir": str(self.output_dir),
            "max_steps": self.max_steps,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "optim": "adamw_torch_fused",
            "bf16": True,
            "fp16": False,
            "tf32": True,
            "gradient_checkpointing": self.gradient_checkpointing,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "use_cache": False,
            "auto_find_batch_size": False,
            "logging_strategy": "steps",
            "logging_steps": self.logging_steps,
            "save_strategy": "steps",
            "save_steps": cadence.save_steps,
            "save_total_limit": cadence.save_total_limit,
            "save_only_model": False,
            "restore_callback_states_from_checkpoint": True,
            "report_to": "none",
            "seed": self.seed,
            "data_seed": self.data_seed,
            "do_train": True,
            "push_to_hub": False,
        }


@dataclass(frozen=True, slots=True)
class SFTRunConfig:
    run: TrainingRunConfig
    max_length: int = 2048
    packing: bool = False
    assistant_only_loss: bool = False
    completion_only_loss: bool | None = True
    checkpoint_save_steps: int | None = None
    checkpoint_save_total_limit: int | None = None

    def __post_init__(self) -> None:
        if self.max_length < 1:
            raise ValueError("max_length must be positive")
        if self.assistant_only_loss and self.completion_only_loss:
            raise ValueError("assistant-only and completion-only loss cannot both be enabled")
        if self.checkpoint_save_steps is not None and not (
            1 <= self.checkpoint_save_steps <= self.run.max_steps
        ):
            raise ValueError("SFT checkpoint save steps must fall within the run")
        if self.checkpoint_save_total_limit is not None and self.checkpoint_save_total_limit < 1:
            raise ValueError("SFT checkpoint retention must be positive")


@dataclass(frozen=True, slots=True)
class GRPORunConfig:
    run: TrainingRunConfig
    num_generations: int = 8
    max_completion_length: int = 512
    beta: float = 0.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float | None = None
    epsilon: float = 0.2
    num_iterations: int = 1
    generation_batch_size: int = 8
    mask_truncated_completions: bool = False
    shuffle_dataset: bool = False
    enable_jit_checkpoint: bool = False
    enable_thinking: bool = False
    checkpoint_save_steps: int | None = None
    checkpoint_save_total_limit: int | None = None
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "linear"
    use_vllm: bool = False
    loss_type: str = "grpo"

    def __post_init__(self) -> None:
        if self.num_generations < 2:
            raise ValueError("GRPO requires at least two generations per prompt")
        if self.max_completion_length < 1:
            raise ValueError("max_completion_length must be positive")
        if self.beta < 0.0:
            raise ValueError("GRPO beta cannot be negative")
        if self.temperature <= 0.0:
            raise ValueError("GRPO temperature must be positive")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("GRPO top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("GRPO top_k cannot be negative")
        if self.min_p is not None and not 0.0 <= self.min_p <= 1.0:
            raise ValueError("GRPO min_p must be in [0, 1]")
        if not 0.0 < self.epsilon < 1.0:
            raise ValueError("GRPO epsilon must be in (0, 1)")
        if self.num_iterations < 1:
            raise ValueError("GRPO num_iterations must be positive")
        if self.generation_batch_size < self.num_generations:
            raise ValueError("generation batch must contain at least one complete GRPO group")
        if self.generation_batch_size % self.num_generations:
            raise ValueError("generation batch must be divisible by num_generations")
        if self.generation_batch_size % self.run.per_device_train_batch_size:
            raise ValueError("generation batch must be divisible by the device microbatch")
        steps_per_generation = self.generation_batch_size // self.run.per_device_train_batch_size
        if self.run.gradient_accumulation_steps % steps_per_generation:
            raise ValueError(
                "gradient accumulation must be divisible into complete generation batches"
            )
        if self.enable_jit_checkpoint:
            raise ValueError(
                "signal checkpointing is not qualified on the canonical Windows backend"
            )
        if self.enable_thinking:
            raise ValueError("the frozen Qwen RL prompt protocol disables thinking")
        if self.checkpoint_save_steps is not None and not (
            1 <= self.checkpoint_save_steps <= self.run.max_steps
        ):
            raise ValueError("GRPO checkpoint_save_steps is outside the run budget")
        if self.checkpoint_save_total_limit is not None and self.checkpoint_save_total_limit < 1:
            raise ValueError("GRPO checkpoint retention must be positive")
        if not 0.0 <= self.adam_beta1 < 1.0 or not 0.0 <= self.adam_beta2 < 1.0:
            raise ValueError("GRPO Adam beta values must be in [0, 1)")
        if self.adam_epsilon <= 0.0 or self.max_grad_norm <= 0.0:
            raise ValueError("GRPO optimizer epsilon and gradient norm must be positive")
        if self.lr_scheduler_type != "linear":
            raise ValueError("the frozen GRPO learning-rate scheduler must be linear")
        if self.use_vllm:
            raise ValueError(
                "HF generation is the qualified correctness reference; vLLM is not enabled"
            )
        if self.loss_type != "grpo":
            raise ValueError("the primary sampler comparison must use standard GRPO loss")
        if self.run.global_batch_size % self.num_generations:
            raise ValueError("single-GPU global batch must be divisible by num_generations")
