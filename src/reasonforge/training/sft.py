"""Local wrapper around the pinned TRL SFT construction API."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from typing import Any

from reasonforge.models import (
    build_lora_config,
    build_qlora_model_kwargs,
    verify_training_library_versions,
)
from reasonforge.training.config import SFTRunConfig


def construct_sft_trainer(
    config: SFTRunConfig,
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
) -> Any:
    """Construct, but do not execute, a pinned SFT trainer."""
    verify_training_library_versions()
    if trainer_cls is None or args_cls is None:
        trl = import_module("trl")
        trainer_cls = trainer_cls or trl.SFTTrainer
        args_cls = args_cls or trl.SFTConfig
    model_kwargs = build_qlora_model_kwargs(
        config.run.model,
        bitsandbytes_config_cls=bitsandbytes_config_cls,
        dtype=dtype,
    )
    quantization_config = model_kwargs.pop("quantization_config")
    trainer_kwargs = config.run.common_trainer_kwargs()
    if config.checkpoint_save_steps is not None:
        trainer_kwargs["save_steps"] = config.checkpoint_save_steps
    if config.checkpoint_save_total_limit is not None:
        trainer_kwargs["save_total_limit"] = config.checkpoint_save_total_limit
    args = args_cls(
        **trainer_kwargs,
        model_init_kwargs=model_kwargs,
        max_length=config.max_length,
        packing=config.packing,
        assistant_only_loss=config.assistant_only_loss,
        completion_only_loss=config.completion_only_loss,
    )
    return trainer_cls(
        model=config.run.model.model_id,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processing_class,
        callbacks=list(callbacks),
        quantization_config=quantization_config,
        peft_config=build_lora_config(config.run.adapter, lora_config_cls=lora_config_cls),
    )
