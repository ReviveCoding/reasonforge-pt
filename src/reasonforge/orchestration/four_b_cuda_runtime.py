"""E36 runtime boundary and bounded real-CUDA qualification implementation.

Imports of torch/transformers are confined to the explicitly invoked real-CUDA
qualification backend.  Scientific C0/C1/C2 execution additionally requires a frozen
runtime recipe contract; missing details are never inferred from labels.
"""

from __future__ import annotations

import gc
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from reasonforge.checkpoints.hashing import atomic_write_json, sha256_json
from reasonforge.systems import GpuOwnerLease

from .four_b_cuda_worker import E36CudaOutOfMemory, RuntimeOutcome, WorkerRequest
from .four_b_transfer import (
    E36_CONFIG,
    E36_MODEL,
    E36_PROTOCOL,
    E36_REVISION,
    FourBIntegrityError,
    _binding,
    _config,
    _load_json,
    _mapping,
    _safe_path,
    _sequence,
    verify_e36_preflight,
)

E36_FROZEN_RUNTIME_RECIPE = "evidence/training/E36/FROZEN_RUNTIME_RECIPE.json"
E36_RUNTIME_CPU_QUALIFICATION = "evidence/training/E36/CUDA_RUNTIME_CPU_QUALIFICATION.json"
E36_RUNTIME_QUALIFICATION_ROOT = "evidence/training/E36/runtime_qualification"


@dataclass(frozen=True, slots=True)
class RuntimeTelemetry:
    cuda_available: bool
    cuda_device_count: int
    cuda_device_name: str
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    model_instances_created: int
    child_model_processes_created: int
    repeated_batch_allocated_bytes: tuple[int, ...]


class RuntimeBackend(Protocol):
    def load_exact_model(self, request: WorkerRequest) -> None: ...

    def run_reference(self, request: WorkerRequest) -> None: ...

    def run_training(
        self, request: WorkerRequest, recipe: Mapping[str, Any]
    ) -> tuple[int, Mapping[str, str | int]]: ...

    def telemetry(self) -> RuntimeTelemetry: ...

    def protected_outputs_accessed(self) -> bool: ...

    def close(self) -> None: ...


def _fallback_expectation(request: WorkerRequest) -> None:
    expected = {
        "LOWER_MICROBATCH_TO_ONE": (0, 1, True, False, "FROZEN_PRIMARY"),
        "INCREASE_GRADIENT_ACCUMULATION_PRESERVING_GLOBAL_WORK": (
            1,
            1,
            True,
            False,
            "FROZEN_PRIMARY",
        ),
        "ENABLE_GRADIENT_CHECKPOINTING": (2, 1, True, True, "FROZEN_PRIMARY"),
        "USE_QUALIFIED_QLORA": (3, 1, True, True, "QUALIFIED_QLORA"),
    }.get(request.fallback_step)
    observed = (
        request.fallback_index,
        request.microbatch_size,
        request.preserve_frozen_global_work,
        request.gradient_checkpointing,
        request.quantization_mode,
    )
    if expected is None or observed != expected:
        raise FourBIntegrityError("E36 runtime fallback profile changed")


def _validate_request(request: WorkerRequest) -> None:
    if (
        request.model_id != E36_MODEL
        or request.model_revision != E36_REVISION
        or Path(request.config_path).name != Path(E36_CONFIG).name
        or request.recipe_id not in {"C0", "C1", "C2"}
        or request.completed_optimizer_step_only is not True
        or request.partial_rollout_groups_excluded is not True
        or request.protected_access_prohibited is not True
    ):
        raise FourBIntegrityError("E36 runtime request identity is invalid")
    expected = {
        "C0": ("frozen_reference", None, None, None),
        "C1": ("strongest_conventional_finalist", "E16", "uniform", "terminal"),
        "C2": ("rcac_finalist", "E18", "rcac", "selected_from_E16"),
    }[request.recipe_id]
    if (
        request.recipe_role,
        request.source_experiment,
        request.sampler,
        request.reward_architecture,
    ) != expected:
        raise FourBIntegrityError(f"E36 runtime {request.recipe_id} boundary changed")
    _fallback_expectation(request)


def _validate_telemetry(telemetry: RuntimeTelemetry) -> None:
    if (
        telemetry.cuda_available is not True
        or telemetry.cuda_device_count != 1
        or telemetry.cuda_device_name != "NVIDIA GeForce RTX 4090 Laptop GPU"
        or telemetry.peak_allocated_bytes <= 0
        or telemetry.peak_reserved_bytes < telemetry.peak_allocated_bytes
        or telemetry.model_instances_created != 1
        or telemetry.child_model_processes_created != 0
    ):
        raise FourBIntegrityError("E36 runtime telemetry violates the single-GPU contract")
    samples = telemetry.repeated_batch_allocated_bytes
    if len(samples) < 3 or any(sample <= 0 for sample in samples):
        raise FourBIntegrityError("E36 runtime lacks repeated-batch memory evidence")
    tolerance = max(64 * 1024**2, int(samples[0] * 0.05))
    if max(samples[1:]) - min(samples[1:]) > tolerance:
        raise FourBIntegrityError("E36 repeated-batch CUDA memory is not bounded")


def execute_with_backend(
    request: WorkerRequest,
    *,
    backend: RuntimeBackend,
    recipe_contract: Mapping[str, Any],
) -> RuntimeOutcome:
    """Execute an exact frozen recipe through a lifecycle-observable backend."""

    _validate_request(request)
    recipes = _mapping(recipe_contract.get("recipes"), "runtime recipes")
    recipe = _mapping(recipes.get(request.recipe_id), f"runtime {request.recipe_id} recipe")
    if (
        recipe_contract.get("schema_version") != 1
        or recipe_contract.get("status") != "FROZEN"
        or recipe_contract.get("experiment_id") != "E36"
        or recipe_contract.get("protocol_id") != E36_PROTOCOL
        or recipe_contract.get("model_id") != E36_MODEL
        or recipe_contract.get("model_revision") != E36_REVISION
        or recipe_contract.get("protected_roles_allowed") != ["QUALIFICATION"]
        or recipe_contract.get("protected_outputs_accessed") is not False
        or set(recipes) != {"C0", "C1", "C2"}
        or recipe.get("role") != request.recipe_role
        or recipe.get("source_experiment") != request.source_experiment
        or recipe.get("sampler") != request.sampler
        or recipe.get("reward_architecture") != request.reward_architecture
    ):
        raise FourBIntegrityError("E36 frozen runtime recipe contract is invalid")
    loaded = False
    try:
        backend.load_exact_model(request)
        loaded = True
        if request.recipe_id == "C0":
            backend.run_reference(request)
            steps = 0
            checkpoint: Mapping[str, str | int] | None = None
            optimizer_created = False
        else:
            steps, checkpoint = backend.run_training(request, recipe)
            if steps <= 0:
                raise FourBIntegrityError("E36 training backend completed no optimizer step")
            optimizer_created = True
        telemetry = backend.telemetry()
        _validate_telemetry(telemetry)
        if backend.protected_outputs_accessed():
            raise FourBIntegrityError("E36 runtime accessed a protected output during execution")
        return RuntimeOutcome(
            cuda_available=telemetry.cuda_available,
            cuda_device_name=telemetry.cuda_device_name,
            peak_allocated_bytes=telemetry.peak_allocated_bytes,
            model_instances_created=telemetry.model_instances_created,
            child_model_processes_created=telemetry.child_model_processes_created,
            optimizer_created=optimizer_created,
            completed_optimizer_steps=steps,
            final_checkpoint_completion=checkpoint,
            resumed_from=request.resume_from,
            protected_outputs_accessed=False,
        )
    finally:
        if loaded:
            backend.close()


def _load_frozen_runtime_recipe(root: Path) -> Mapping[str, Any]:
    value = _load_json(
        _safe_path(root, E36_FROZEN_RUNTIME_RECIPE), "frozen runtime recipe contract"
    )
    config = _mapping(value.get("configuration"), "runtime recipe configuration")
    if dict(config) != _binding(root, E36_CONFIG):
        raise FourBIntegrityError("E36 runtime recipe does not bind the frozen E36 config")
    return value


def execute_e36_request(request: WorkerRequest) -> RuntimeOutcome:
    """Production entry point; fail closed until an exact runtime recipe is frozen."""

    root = Path(request.workspace).resolve()
    _load_frozen_runtime_recipe(root)
    raise FourBIntegrityError(
        "E36 scientific C0/C1/C2 backend is not qualified; the bounded C0 operational "
        "fixture must not substitute for transfer execution"
    )


class TransformersQualificationBackend:
    """HF correctness-reference backend used only by an explicit CUDA invocation."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self._torch: Any = None
        self._model: Any = None
        self._tokenizer: Any = None
        self._telemetry: RuntimeTelemetry | None = None
        self._model_instances = 0

    def load_exact_model(self, request: WorkerRequest) -> None:
        if request.recipe_id != "C0":
            raise FourBIntegrityError(
                "E36 Transformers qualification backend cannot substitute for C1/C2 training"
            )
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as error:  # pragma: no cover - real qualification only
            raise FourBIntegrityError("E36 CUDA runtime dependencies are unavailable") from error
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise FourBIntegrityError("E36 real qualification requires exactly one CUDA GPU")
        if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 4090 Laptop GPU":
            raise FourBIntegrityError("E36 CUDA device identity changed")
        if not torch.cuda.is_bf16_supported():
            raise FourBIntegrityError("E36 BF16 support is unavailable")
        kwargs: dict[str, Any] = {
            "revision": E36_REVISION,
            "local_files_only": True,
            "cache_dir": self.root / ".cache/huggingface/hub",
            "device_map": {"": 0},
            "trust_remote_code": False,
        }
        if request.quantization_mode == "QUALIFIED_QLORA":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["torch_dtype"] = torch.bfloat16
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                E36_MODEL,
                revision=E36_REVISION,
                local_files_only=True,
                cache_dir=self.root / ".cache/huggingface/hub",
                trust_remote_code=False,
                padding_side="left",
            )
            self._model = AutoModelForCausalLM.from_pretrained(E36_MODEL, **kwargs)
        except torch.OutOfMemoryError as error:
            raise E36CudaOutOfMemory(
                device_name=torch.cuda.get_device_name(0),
                peak_allocated_bytes=max(1, int(torch.cuda.max_memory_allocated(0))),
            ) from error
        self._model_instances += 1
        self._model.eval()
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)
        self._torch = torch

    def run_reference(self, request: WorkerRequest) -> None:
        if self._model is None or self._tokenizer is None or self._torch is None:
            raise FourBIntegrityError("E36 qualification model is not loaded")
        prompt = "Operational qualification only. Return exactly the word READY."
        allocated: list[int] = []
        try:
            for _ in range(3):
                tokens = self._tokenizer(prompt, return_tensors="pt")
                tokens = {key: value.to("cuda:0") for key, value in tokens.items()}
                with self._torch.inference_mode():
                    generated = self._model.generate(
                        **tokens,
                        do_sample=False,
                        max_new_tokens=8,
                        use_cache=True,
                    )
                del generated, tokens
                gc.collect()
                self._torch.cuda.synchronize(0)
                allocated.append(int(self._torch.cuda.memory_allocated(0)))
        except self._torch.OutOfMemoryError as error:
            raise E36CudaOutOfMemory(
                device_name=self._torch.cuda.get_device_name(0),
                peak_allocated_bytes=max(
                    1, int(self._torch.cuda.max_memory_allocated(0))
                ),
            ) from error
        self._telemetry = RuntimeTelemetry(
            cuda_available=True,
            cuda_device_count=1,
            cuda_device_name=self._torch.cuda.get_device_name(0),
            peak_allocated_bytes=int(self._torch.cuda.max_memory_allocated(0)),
            peak_reserved_bytes=int(self._torch.cuda.max_memory_reserved(0)),
            model_instances_created=self._model_instances,
            child_model_processes_created=0,
            repeated_batch_allocated_bytes=tuple(allocated),
        )

    def run_training(
        self, request: WorkerRequest, recipe: Mapping[str, Any]
    ) -> tuple[int, Mapping[str, str | int]]:
        del request, recipe
        raise FourBIntegrityError(
            "E36 C1/C2 training runtime is not encoded; refusing to infer a scientific recipe"
        )

    def telemetry(self) -> RuntimeTelemetry:
        if self._telemetry is None:
            raise FourBIntegrityError("E36 runtime telemetry is unavailable")
        return self._telemetry

    def protected_outputs_accessed(self) -> bool:
        return False

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        if self._torch is not None:
            gc.collect()
            self._torch.cuda.empty_cache()
            self._torch.cuda.synchronize(0)


def run_bounded_cuda_qualification(root: Path, *, fallback_step: str) -> dict[str, Any]:
    """Run one explicit C0 model-load/repeated-batch qualification under the GPU lease."""

    root = root.resolve()
    config, config_sha256 = _config(root)
    preflight = verify_e36_preflight(root)
    if preflight.get("status") != "READY_FOR_BOUNDED_CUDA_QUALIFICATION":
        raise FourBIntegrityError("E36 real CUDA qualification requires a ready preflight")
    ladder = [str(item) for item in _sequence(config.get("authorized_fallback_ladder"), "ladder")]
    if fallback_step not in ladder:
        raise FourBIntegrityError("E36 real CUDA qualification fallback is unauthorized")
    profile = {
        "fallback_index": ladder.index(fallback_step),
        "microbatch_size": 1,
        "preserve_frozen_global_work": True,
        "gradient_checkpointing": ladder.index(fallback_step) >= 2,
        "quantization_mode": (
            "QUALIFIED_QLORA" if ladder.index(fallback_step) >= 3 else "FROZEN_PRIMARY"
        ),
    }
    request = WorkerRequest(
        workspace=str(root),
        config_path=str(_safe_path(root, E36_CONFIG)),
        recipe_id="C0",
        recipe_role="frozen_reference",
        source_experiment=None,
        sampler=None,
        reward_architecture=None,
        run_id="E36-runtime-qualification-c0-v1",
        run_identity_hash=sha256_json(
            {"protocol_id": E36_PROTOCOL, "fallback_step": fallback_step}
        ),
        model_id=E36_MODEL,
        model_revision=E36_REVISION,
        fallback_step=fallback_step,
        resume_from=None,
        completed_optimizer_step_only=True,
        partial_rollout_groups_excluded=True,
        protected_access_prohibited=True,
        **profile,
    )
    contract = {
        "schema_version": 1,
        "status": "FROZEN",
        "experiment_id": "E36",
        "protocol_id": E36_PROTOCOL,
        "model_id": E36_MODEL,
        "model_revision": E36_REVISION,
        "protected_roles_allowed": ["QUALIFICATION"],
        "protected_outputs_accessed": False,
        "recipes": {
            "C0": {
                "role": "frozen_reference",
                "source_experiment": None,
                "sampler": None,
                "reward_architecture": None,
            },
            "C1": {
                "role": "strongest_conventional_finalist",
                "source_experiment": "E16",
                "sampler": "uniform",
                "reward_architecture": "terminal",
            },
            "C2": {
                "role": "rcac_finalist",
                "source_experiment": "E18",
                "sampler": "rcac",
                "reward_architecture": "selected_from_E16",
            },
        },
    }
    with GpuOwnerLease(root, "E36", f"runtime-qualification-{fallback_step}"):
        outcome = execute_with_backend(
            request, backend=TransformersQualificationBackend(root), recipe_contract=contract
        )
    return {
        "schema_version": 1,
        "status": "QUALIFIED",
        "qualification_scope": "REAL_CUDA_QWEN3_4B_C0_PROFILE",
        "experiment_id": "E36",
        "protocol_id": E36_PROTOCOL,
        "model_id": E36_MODEL,
        "model_revision": E36_REVISION,
        "fallback_step": fallback_step,
        "request_identity": sha256_json(asdict(request)),
        "configuration": _binding(root, E36_CONFIG),
        "config_sha256": config_sha256,
        "preflight": _binding(root, "evidence/training/E36/FOUR_B_PREFLIGHT.json"),
        "outcome": asdict(outcome),
        "single_model_instance": True,
        "single_gpu_process": True,
        "protected_outputs_accessed": False,
    }


def publish_bounded_cuda_qualification(
    root: Path, *, fallback_step: str, value: Mapping[str, Any]
) -> Path:
    """Publish one already executed profile without finalizing runtime qualification."""

    root = root.resolve()
    if (
        value.get("status") != "QUALIFIED"
        or value.get("fallback_step") != fallback_step
        or value.get("protected_outputs_accessed") is not False
    ):
        raise FourBIntegrityError("E36 CUDA qualification result is invalid")
    destination = _safe_path(
        root, f"{E36_RUNTIME_QUALIFICATION_ROOT}/{fallback_step}.json"
    )
    if destination.exists():
        raise FourBIntegrityError("E36 CUDA qualification result already exists")
    atomic_write_json(destination, dict(value))
    return destination
