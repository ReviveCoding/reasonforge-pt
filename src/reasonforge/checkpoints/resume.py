"""Read-only resume planning; dry runs never mutate project state."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from .models import CheckpointIntegrityError, RunIdentity
from .project import ProjectCheckpointStore
from .runtime import RuntimeCheckpointStore

ResumeAction = Literal[
    "START_NEW",
    "NEW_RUN_ID_REQUIRED",
    "RESUME",
    "ADVANCE_DAG",
    "BLOCKED_INTEGRITY",
]


@dataclass(frozen=True)
class ResumePlan:
    action: ResumeAction
    reason: str
    project_checkpoint_id: str | None = None
    experiment_id: str | None = None
    run_id: str | None = None
    runtime_checkpoint: str | None = None
    checkpoint_step: int | None = None
    next_authorized_action: str | None = None
    mutates_state: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def dry_run_resume(
    project_store: ProjectCheckpointStore,
    runtime_store: RuntimeCheckpointStore,
    *,
    experiment_id: str | None = None,
    run_id: str | None = None,
    expected_identity: RunIdentity | None = None,
) -> ResumePlan:
    """Compute the exact legal recovery action without writing any files."""

    if not project_store.latest_path.exists():
        return ResumePlan(
            action="START_NEW", reason="no verified project checkpoint pointer exists"
        )
    try:
        project, _ = project_store.verify_latest()
    except CheckpointIntegrityError as error:
        return ResumePlan(action="BLOCKED_INTEGRITY", reason=str(error))
    if experiment_id is None or run_id is None or expected_identity is None:
        return ResumePlan(
            action="ADVANCE_DAG",
            reason="project checkpoint verified; no interrupted run was specified",
            project_checkpoint_id=project.checkpoint_id,
            next_authorized_action=project.next_authorized_action,
        )
    compatible = runtime_store.latest_compatible(experiment_id, run_id, expected_identity)
    if compatible is None:
        complete_root = runtime_store.root / experiment_id / run_id / "COMPLETE"
        has_existing_complete = complete_root.is_dir() and any(
            path.is_dir() for path in complete_root.glob("checkpoint_*")
        )
        return ResumePlan(
            action="NEW_RUN_ID_REQUIRED" if has_existing_complete else "START_NEW",
            reason=(
                "existing runtime checkpoints are incompatible; a new run_id is required"
                if has_existing_complete
                else "no complete compatible runtime checkpoint exists"
            ),
            experiment_id=experiment_id,
            run_id=run_id,
            project_checkpoint_id=project.checkpoint_id,
            next_authorized_action=project.next_authorized_action,
        )
    path, runtime = compatible
    return ResumePlan(
        action="RESUME",
        reason="latest complete compatible runtime checkpoint verified",
        experiment_id=experiment_id,
        run_id=run_id,
        runtime_checkpoint=str(path),
        checkpoint_step=runtime.checkpoint_step,
        project_checkpoint_id=project.checkpoint_id,
        next_authorized_action=project.next_authorized_action,
    )
