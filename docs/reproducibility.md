# Reproducibility

## Public CPU layer

Python 3.11 is the supported CI version. From the repository root:

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m pytest -q
python scripts/validate_public_release.py
```

This layer checks exported sampler/statistics interfaces, portable result claims, CSV schemas, SVG validity, and local artifact links. It does not recreate model outputs or assert that the public summary is the original CP110 manifest.

## Full GPU research layer

The frozen model identity is `Qwen/Qwen3-4B` at revision `1cfa9a7208912126459214e8b04321603b3df60c`. The common 4B initialization used QLoRA with NF4 double quantization and BF16 compute, a fresh adapter with rank 8, alpha 16, dropout 0.05, and `q_proj`/`v_proj` targets. The reference SFT stage used 60 steps. The matched C1/C2 GRPO stage used 32 optimizer steps, 64 rollout groups, and 512 total rollouts. The actual frozen configs under [`configs/`](../configs/) provide the detailed research settings.

An end-to-end reproduction requires compatible CUDA hardware (the recorded environment used one NVIDIA RTX 4090 Laptop GPU), exact model and dataset revisions, training and locked-evaluation pools, deterministic scorers, the common SFT adapter, and sufficient storage for model/runtime artifacts. These inputs are not present in this portfolio repository. Dataset and model access terms must be checked independently before acquisition.

Conceptual sequence:

1. Acquire and verify pinned model and licensed data under their own terms.
2. Train the frozen common SFT/QLoRA adapter.
3. Run matched C1 uniform and C2 RCAC post-training for seeds 1701, 1702, and 1703.
4. Freeze evaluation inputs and score both arms on identical stable example IDs.
5. Apply the preregistered +0.03 promotion gate to the locked primary endpoint.
6. Run separate post-final diagnostics only after the decision is sealed.

These are protocol steps, not runnable public CI commands. The original evidence repositories contain the execution manifests and forward-only repair chain. The public export intentionally omits model weights, adapters, optimizer states, raw predictions, raw telemetry, benchmark payloads, and machine-local paths. This keeps CI CPU-safe and avoids presenting an incomplete GPU reproduction as executable.

Internal analytical completion: `CP110_POST_CP100_ANALYTICS_COMPLETE`. The [public release summary](../evidence/release/PUBLIC_RELEASE_SUMMARY.json) is a derived portable artifact, not a replacement checkpoint.
