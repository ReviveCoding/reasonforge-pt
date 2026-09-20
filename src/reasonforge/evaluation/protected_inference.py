"""CUDA-only, bounded-memory predictor used by the protected protocol runner.

Heavy dependencies are imported lazily so protocol and scorer tests remain CPU-safe.
"""

from __future__ import annotations

import gc
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .protected_protocol import GenerationMode, ProtectedGenerationExample


class CudaProtectedPredictor:
    """Load one frozen QLoRA model and process exactly one example at a time."""

    def __init__(
        self,
        *,
        base_model_id: str,
        base_revision: str,
        adapter_path: Path | None,
        max_new_tokens: int,
        use_cache: bool,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError("protected neural evaluation requires CUDA")
        self._torch = torch
        self.max_new_tokens = max_new_tokens
        self.use_cache = use_cache
        self.model_construction_count = 0
        self.adapter_wrapper_count = 0
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_id,
            revision=base_revision,
            local_files_only=True,
            padding_side="left",
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            revision=base_revision,
            local_files_only=True,
            quantization_config=quantization,
            device_map={"": 0},
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        self.model_construction_count += 1
        if adapter_path is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
            self.adapter_wrapper_count += 1
        model.requires_grad_(False)
        model.eval()
        trainable = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        loaded_in_4bit = bool(
            getattr(model, "is_loaded_in_4bit", False)
            or getattr(getattr(model, "base_model", None), "is_loaded_in_4bit", False)
            or getattr(
                getattr(getattr(model, "base_model", None), "model", None),
                "is_loaded_in_4bit",
                False,
            )
        )
        if model.training or trainable != 0 or not loaded_in_4bit:
            raise RuntimeError(
                "protected model must be eval-only with frozen gradients and one 4-bit base"
            )
        self.model = model
        self.tokenizer = tokenizer
        self.stop_token_ids = sorted(
            {
                int(tokenizer.eos_token_id),
                int(tokenizer.convert_tokens_to_ids("<|im_end|>")),
            }
        )

    def diagnostics(self) -> dict[str, Any]:
        torch = self._torch
        parameters = list(self.model.parameters())
        return {
            "model_construction_count": self.model_construction_count,
            "adapter_wrapper_count": self.adapter_wrapper_count,
            "eval_mode": not bool(self.model.training),
            "grad_enabled": torch.is_grad_enabled(),
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in parameters if parameter.requires_grad
            ),
            "optimizer_constructed": False,
            "parameter_devices": dict(Counter(str(item.device) for item in parameters)),
            "batch_size": 1,
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
        }

    def _render(self, example: ProtectedGenerationExample) -> str:
        kwargs: dict[str, Any] = {}
        if example.tool_schema is not None:
            kwargs["tools"] = json.loads(example.tool_schema)
        return str(
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": example.prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
                **kwargs,
            )
        )

    def __call__(
        self, example: ProtectedGenerationExample, mode: GenerationMode
    ) -> tuple[str, dict[str, Any]]:
        torch = self._torch
        torch.cuda.reset_peak_memory_stats()
        if mode == "generate":
            prompt = self._render(example)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    use_cache=self.use_cache,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.stop_token_ids,
                )
            prompt_length = int(inputs["input_ids"].shape[1])
            completion_ids = generated[:, prompt_length:].detach().cpu()
            output = str(self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)[0])
            del generated, completion_ids, inputs
        else:
            if not example.candidates:
                raise ValueError("conditional log-probability example has no candidates")
            prompt = self._render(example)
            prompt_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)[
                "input_ids"
            ].to(self.model.device)
            candidate_scores: dict[str, float] = {}
            with torch.inference_mode():
                for candidate in example.candidates:
                    candidate_ids = self.tokenizer(
                        " " + candidate, return_tensors="pt", add_special_tokens=False
                    )["input_ids"].to(self.model.device)
                    input_ids = torch.cat((prompt_ids, candidate_ids), dim=1)
                    logits = self.model(input_ids=input_ids, use_cache=False).logits
                    start = int(prompt_ids.shape[1]) - 1
                    token_logits = logits[:, start:-1, :]
                    log_probs = torch.log_softmax(token_logits.float(), dim=-1)
                    gathered = log_probs.gather(-1, candidate_ids.unsqueeze(-1)).squeeze(-1)
                    digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
                    candidate_scores[digest] = float(gathered.sum().item())
                    del candidate_ids, input_ids, logits, token_logits, log_probs, gathered
            output = json.dumps(
                {"candidate_logprob": candidate_scores},
                sort_keys=True,
                separators=(",", ":"),
            )
            del prompt_ids
        torch.cuda.synchronize()
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return output, {
            "finish_reason": "scored" if mode == "conditional_logprob" else "stop",
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "post_cleanup_allocated_bytes": int(torch.cuda.memory_allocated()),
            "post_cleanup_reserved_bytes": int(torch.cuda.memory_reserved()),
        }
