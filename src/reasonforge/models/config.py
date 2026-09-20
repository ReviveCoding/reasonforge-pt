"""Frozen QLoRA/model configuration bound to the qualified library stack."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any

EXPECTED_LIBRARY_VERSIONS = {
    "transformers": "5.16.1",
    "trl": "1.12.0",
    "peft": "0.20.0",
}


class LibraryVersionMismatch(RuntimeError):
    """The installed training stack differs from the qualified pins."""


def verify_training_library_versions(
    version_getter: Callable[[str], str] = version,
) -> dict[str, str]:
    """Fail closed when a scientifically material training package is unpinned."""
    observed: dict[str, str] = {}
    for distribution, expected in EXPECTED_LIBRARY_VERSIONS.items():
        try:
            installed = version_getter(distribution)
        except PackageNotFoundError as exc:
            raise LibraryVersionMismatch(
                f"required package is not installed: {distribution}"
            ) from exc
        observed[distribution] = installed
        if installed != expected:
            raise LibraryVersionMismatch(
                f"{distribution} version mismatch: expected {expected}, observed {installed}"
            )
    return observed


class AdapterTarget(StrEnum):
    QV = "qv"
    ALL_ATTENTION = "all_attention"
    ATTENTION_MLP = "attention_mlp"
    ALL_LINEAR = "all_linear"


_TARGET_MODULES: dict[AdapterTarget, tuple[str, ...] | str] = {
    AdapterTarget.QV: ("q_proj", "v_proj"),
    AdapterTarget.ALL_ATTENTION: ("q_proj", "k_proj", "v_proj", "o_proj"),
    AdapterTarget.ATTENTION_MLP: (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ),
    AdapterTarget.ALL_LINEAR: "all-linear",
}


@dataclass(frozen=True, slots=True)
class QLoRAModelConfig:
    model_id: str
    revision: str
    compute_dtype: str = "bfloat16"
    quant_type: str = "nf4"
    use_double_quant: bool = True
    attn_implementation: str = "sdpa"
    device_index: int = 0
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        if not self.model_id.strip() or not self.revision.strip():
            raise ValueError("model_id and immutable revision are required")
        if self.compute_dtype != "bfloat16":
            raise ValueError("the qualified QLoRA path requires bfloat16 compute")
        if self.quant_type != "nf4":
            raise ValueError("the qualified QLoRA path requires NF4 quantization")
        if self.attn_implementation != "sdpa":
            raise ValueError("the qualified attention implementation is SDPA")
        if self.device_index < 0:
            raise ValueError("device_index cannot be negative")


@dataclass(frozen=True, slots=True)
class LoRAAdapterConfig:
    target: AdapterTarget = AdapterTarget.ATTENTION_MLP
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    use_rslora: bool = False
    bias: str = "none"

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("LoRA rank must be positive")
        if self.alpha < 1:
            raise ValueError("LoRA alpha must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if self.bias not in {"none", "all", "lora_only"}:
            raise ValueError("unsupported LoRA bias mode")

    @property
    def target_modules(self) -> tuple[str, ...] | str:
        return _TARGET_MODULES[self.target]


def build_qlora_model_kwargs(
    config: QLoRAModelConfig,
    *,
    bitsandbytes_config_cls: type[Any] | None = None,
    dtype: object | None = None,
) -> dict[str, Any]:
    """Construct explicit CUDA QLoRA kwargs without loading a model."""
    if bitsandbytes_config_cls is None or dtype is None:
        verify_training_library_versions()
    if dtype is None:
        dtype = import_module("torch").bfloat16
    if bitsandbytes_config_cls is None:
        bitsandbytes_config_cls = import_module("transformers").BitsAndBytesConfig
    quantization = bitsandbytes_config_cls(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.quant_type,
        bnb_4bit_use_double_quant=config.use_double_quant,
        bnb_4bit_compute_dtype=dtype,
    )
    return {
        "revision": config.revision,
        "dtype": dtype,
        "device_map": {"": config.device_index},
        "attn_implementation": config.attn_implementation,
        "trust_remote_code": config.trust_remote_code,
        "use_cache": False,
        "quantization_config": quantization,
    }


def build_lora_config(
    config: LoRAAdapterConfig,
    *,
    lora_config_cls: type[Any] | None = None,
) -> Any:
    """Construct the PEFT LoRA configuration for causal language modeling."""
    if lora_config_cls is None:
        verify_training_library_versions()
        lora_config_cls = import_module("peft").LoraConfig
    targets = config.target_modules
    return lora_config_cls(
        task_type="CAUSAL_LM",
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=list(targets) if isinstance(targets, tuple) else targets,
        bias=config.bias,
        use_rslora=config.use_rslora,
    )
