# Methodology

ReasonForge-PT compares task-selection policies under a shared post-training recipe. The 4B study begins with one frozen Qwen3-4B SFT/QLoRA adapter. C1 uses uniform task sampling; C2 uses the previously selected Reliability-Constrained Adaptive Curriculum (RCAC) sampler. RCAC changes task selection, not the GRPO policy-gradient estimator. The two arms share the source model, starting adapter, training pool, optimizer, reward architecture, rollout budget, and evaluation configuration.

The representative frozen recipe is in [`configs/e39/original_common_recipe_v1.yaml`](../configs/e39/original_common_recipe_v1.yaml); the C1/C2 fairness contract is in [`configs/analysis/c1_c2_fairness_contract_v2.yaml`](../configs/analysis/c1_c2_fairness_contract_v2.yaml). These copied configs are research references. Data files, adapters, and runtime manifests needed to execute them are intentionally absent from this public export.

The locked primary endpoint is the matched C2-minus-C1 task-score difference across 150 stable example identities in each of three seeds. The practical promotion threshold was frozen at +0.03. The final decision used only the preregistered primary comparison. Post-final analyses did not change the threshold, model selection, or E47 decision.

The diagnostic layer examines paired transitions, error modes, slices, perturbation robustness, stochastic rollouts, uncertainty-quality associations, generation-policy sensitivity, grouped routing, and systems telemetry. Its samples and estimands are not interchangeable with the locked primary benchmark. In particular, 450 cells are 150 repeated examples across three seeds, and pooled binary transition counts are descriptive.

The public tables are aggregates selected from the completed analytical release. Their inclusion does not turn post-final diagnostics into promotion evidence. [Claim boundaries](claim_boundaries.md) and [limitations](limitations.md) make that distinction explicit.
