# Licensing and redistribution

The original ReasonForge source package declares **Apache-2.0** in its `pyproject.toml`. The public export carries the standard Apache-2.0 text in [`LICENSE`](../LICENSE) for original ReasonForge code and project-authored documentation. This does not grant rights to third-party model weights, datasets, benchmark payloads, or generated outputs derived from them.

| Material | Public-export disposition |
|---|---|
| Original ReasonForge code | Included under the source project's declared Apache-2.0 license. Public packaging adapts the CPU import surface and adds portable validation. |
| Qwen/Qwen3-4B | Referenced by pinned model ID and revision only. Weights and tokenizer assets are not redistributed; obtain them from the model publisher under its current terms. |
| External benchmarks | Raw rows, prompts, reference answers, and evaluation code copied from benchmark publishers are omitted. The internal register records source-specific terms; this export includes only selected aggregates. |
| Project-generated rows and model outputs | Raw training/evaluation examples, completions, and adapters are omitted because the source register marks redistribution unresolved and model-output rights may depend on other terms. |
| Selected aggregate reports and figures | Included as project-authored derived summaries. They do not reconstruct source benchmark payloads and do not change the frozen scientific result. |

The internal data-license register identifies restricted or unresolved sources, including custom-license Apple benchmarks, gated material, and conflicting-license candidates. None of their raw content is included here. Before any broader redistribution or full reproduction, recheck the exact current model, dataset, and dependency terms. This repository license is deliberately scoped to the original code and public documentation, not to omitted third-party assets.
