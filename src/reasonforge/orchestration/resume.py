"""Repository-level compatibility adapter for read-only resume resolution."""

from __future__ import annotations

from pathlib import Path

from reasonforge.checkpoints import ProjectCheckpointStore, RuntimeCheckpointStore
from reasonforge.checkpoints.resume import dry_run_resume


def resolve_resume(workspace: Path | str, *, dry_run: bool = True) -> dict[str, object]:
    """Resolve the next project action without guessing a scientific run identity.

    The repository-level CLI does not carry a full :class:`RunIdentity`, so it
    deliberately resolves only the verified project milestone and its declared
    next action. Training integrations must call ``dry_run_resume`` with the
    expected identity before invoking Trainer/TRL ``resume_from_checkpoint``.
    """

    root = Path(workspace)
    plan = dry_run_resume(
        ProjectCheckpointStore(root / "evidence" / "checkpoints"),
        RuntimeCheckpointStore(root / "artifacts" / "checkpoints"),
    )
    result = plan.to_dict()
    result["requested_dry_run"] = dry_run
    result["resolution_only"] = True
    result["mutates_state"] = False
    return result
