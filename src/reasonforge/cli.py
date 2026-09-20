"""Command-line entry point for auditable project operations."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import print_json

from reasonforge.config import load_registry
from reasonforge.systems.doctor import collect_environment, write_environment

app = typer.Typer(no_args_is_help=True, help="ReasonForge-PT research orchestrator")
checkpoint_app = typer.Typer(no_args_is_help=True, help="Project checkpoint operations")
app.add_typer(checkpoint_app, name="checkpoint")


@app.command()
def doctor(
    write: Path | None = typer.Option(  # noqa: B008
        None, help="Atomically write the JSON snapshot."
    ),
    no_cuda_exercise: bool = typer.Option(False, help="Do not allocate a CUDA BF16 test tensor."),
) -> None:
    """Inventory the local backend and exercise CUDA when available."""
    snapshot = collect_environment(exercise_cuda=not no_cuda_exercise)
    if write is not None:
        write_environment(snapshot, write)
    print_json(data=snapshot)
    cuda = snapshot.get("cuda", {})
    if not cuda.get("available") or not cuda.get("bf16_matmul_finite", no_cuda_exercise):
        raise typer.Exit(2)


@app.command()
def status(registry: Path = Path("configs/EXPERIMENT_REGISTRY.yaml")) -> None:
    """Show registry state without mutating it."""
    parsed = load_registry(registry)
    counts: dict[str, int] = {}
    for experiment in parsed.experiments:
        counts[experiment.status.value] = counts.get(experiment.status.value, 0) + 1
    print_json(
        data={
            "protocol_id": parsed.protocol_id,
            "frozen": parsed.frozen,
            "counts": counts,
            "experiments": [item.model_dump(mode="json") for item in parsed.experiments],
        }
    )


@app.command("verify")
def verify() -> None:
    """Validate registries and, when present, the latest project checkpoint."""
    registry = load_registry(Path("configs/EXPERIMENT_REGISTRY.yaml"))
    result: dict[str, object] = {"registry": "VALID", "protocol_id": registry.protocol_id}
    latest = Path("evidence/checkpoints/LATEST.json")
    if latest.exists():
        from reasonforge.checkpoints.project import verify_latest

        result["latest_checkpoint"] = verify_latest(Path("evidence/checkpoints"))
    else:
        result["latest_checkpoint"] = "ABSENT"
    print_json(data=json.loads(json.dumps(result, default=str)))


@app.command("resume")
def resume(dry_run: bool = typer.Option(False, help="Report action without mutation.")) -> None:
    """Resolve the next legal DAG action or compatible runtime resume."""
    from reasonforge.orchestration.resume import resolve_resume

    decision = resolve_resume(Path.cwd(), dry_run=dry_run)
    print_json(data=decision)


@checkpoint_app.command("verify")
def checkpoint_verify() -> None:
    """Verify the checkpoint referenced by LATEST."""
    from reasonforge.checkpoints.project import verify_latest

    print_json(data=verify_latest(Path("evidence/checkpoints")))


if __name__ == "__main__":
    app()
