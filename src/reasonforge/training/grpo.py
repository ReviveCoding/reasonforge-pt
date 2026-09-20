"""Local wrapper around standard GRPO in the pinned TRL API."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from typing import Any

from reasonforge.models import (
    build_lora_config,
    build_qlora_model_kwargs,
    verify_training_library_versions,
)
from reasonforge.training.config import GRPORunConfig
from reasonforge.training.runtime import CompletedStepCallback, CompletedStepStateHook


def construct_grpo_trainer(
    config: GRPORunConfig,
    *,
    reward_funcs: object,
    train_dataset: object,
    eval_dataset: object | None = None,
    processing_class: object | None = None,
    callbacks: Sequence[object] = (),
    completed_step_hook: CompletedStepStateHook | None = None,
    tools: list[object] | None = None,
    rollout_func: object | None = None,
    environment_factory: object | None = None,
    model_override: object | None = None,
    continue_existing_adapter: bool = False,
    trainer_cls: type[Any] | None = None,
    args_cls: type[Any] | None = None,
    bitsandbytes_config_cls: type[Any] | None = None,
    lora_config_cls: type[Any] | None = None,
    dtype: object | None = None,
) -> Any:
    """Construct a standard-GRPO trainer without starting rollouts."""
    verify_training_library_versions()
    if trainer_cls is None or args_cls is None:
        trl = import_module("trl")
        trainer_cls = trainer_cls or trl.GRPOTrainer
        args_cls = args_cls or trl.GRPOConfig
    if model_override is not None and not continue_existing_adapter:
        raise ValueError("a preloaded model must explicitly continue the existing adapter")
    if continue_existing_adapter and model_override is None:
        raise ValueError("continuing an existing adapter requires a preloaded model")
    model_kwargs: dict[str, object] = {}
    quantization_config: object | None = None
    if model_override is None:
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
    if model_override is None:
        trainer_kwargs["model_init_kwargs"] = model_kwargs
    args = args_cls(
        **trainer_kwargs,
        num_generations=config.num_generations,
        max_completion_length=config.max_completion_length,
        beta=config.beta,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        min_p=config.min_p,
        epsilon=config.epsilon,
        num_iterations=config.num_iterations,
        adam_beta1=config.adam_beta1,
        adam_beta2=config.adam_beta2,
        adam_epsilon=config.adam_epsilon,
        max_grad_norm=config.max_grad_norm,
        lr_scheduler_type=config.lr_scheduler_type,
        generation_batch_size=config.generation_batch_size,
        mask_truncated_completions=config.mask_truncated_completions,
        shuffle_dataset=config.shuffle_dataset,
        enable_jit_checkpoint=config.enable_jit_checkpoint,
        chat_template_kwargs={"enable_thinking": config.enable_thinking},
        use_vllm=config.use_vllm,
        loss_type=config.loss_type,
        scale_rewards="group",
        multi_objective_aggregation="sum_then_normalize",
        log_completions=True,
    )
    all_callbacks = list(callbacks)
    if completed_step_hook is not None:
        all_callbacks.append(CompletedStepCallback(completed_step_hook))
    constructor_kwargs: dict[str, object] = {
        "model": model_override if model_override is not None else config.run.model.model_id,
        "reward_funcs": reward_funcs,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "processing_class": processing_class,
        "callbacks": all_callbacks,
        "tools": tools,
        "rollout_func": rollout_func,
        "environment_factory": environment_factory,
    }
    if model_override is None:
        constructor_kwargs["quantization_config"] = quantization_config
        constructor_kwargs["peft_config"] = build_lora_config(
            config.run.adapter, lora_config_cls=lora_config_cls
        )
    return trainer_cls(**constructor_kwargs)
