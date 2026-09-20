"""Frozen offline-DPO recipe and wrapper for the pinned TRL API."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from reasonforge.models import (
    build_lora_config,
    build_qlora_model_kwargs,
    verify_training_library_versions,
)
from reasonforge.training.config import TrainingRunConfig


@dataclass(frozen=True, slots=True)
class DPORunConfig:
    run: TrainingRunConfig
    beta: float = 0.1
    loss_type: str = "sigmoid"
    max_length: int = 2048
    label_smoothing: float = 0.0

    def __post_init__(self) -> None:
        if self.beta <= 0.0:
            raise ValueError("DPO beta must be positive")
        if not self.loss_type:
            raise ValueError("DPO loss_type is required")
        if self.max_length < 1:
            raise ValueError("DPO max_length must be positive")
        if not 0.0 <= self.label_smoothing < 0.5:
            raise ValueError("DPO label_smoothing must be in [0, 0.5)")


def construct_dpo_trainer(
    config: DPORunConfig,
    *,
    train_dataset: object,
    eval_dataset: object | None = None,
    processing_class: object | None = None,
    callbacks: Sequence[object] = (),
    trainer_cls: type[Any] | None = None,
    args_cls: type[Any] | None = None,
    bitsandbytes_config_cls: type[Any] | None = None,
    lora_config_cls: type[Any] | None = None,
    dtype: object | None = None,
    model_override: object | None = None,
    continue_existing_adapter: bool = False,
) -> Any:
    """Construct, but do not execute, an offline DPO trainer."""
    verify_training_library_versions()
    if trainer_cls is None or args_cls is None:
        trl = import_module("trl")
        trainer_cls = trainer_cls or trl.DPOTrainer
        args_cls = args_cls or trl.DPOConfig
    if continue_existing_adapter and model_override is None:
        raise ValueError("continuing an adapter requires a preloaded model")
    if model_override is not None and not continue_existing_adapter:
        raise ValueError("a preloaded DPO model must explicitly continue an existing adapter")
    trainer_model: object = config.run.model.model_id
    trainer_extras: dict[str, object] = {}
    args_extras: dict[str, object] = {}
    if model_override is None:
        model_kwargs = build_qlora_model_kwargs(
            config.run.model,
            bitsandbytes_config_cls=bitsandbytes_config_cls,
            dtype=dtype,
        )
        trainer_extras["quantization_config"] = model_kwargs.pop("quantization_config")
        trainer_extras["peft_config"] = build_lora_config(
            config.run.adapter, lora_config_cls=lora_config_cls
        )
        args_extras["model_init_kwargs"] = model_kwargs
    else:
        trainer_model = model_override
    args = args_cls(
        **config.run.common_trainer_kwargs(),
        **args_extras,
        beta=config.beta,
        loss_type=[config.loss_type],
        max_length=config.max_length,
        label_smoothing=config.label_smoothing,
    )
    return trainer_cls(
        model=trainer_model,
        ref_model=None,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processing_class,
        callbacks=list(callbacks),
        **trainer_extras,
    )
