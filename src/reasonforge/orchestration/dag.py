"""Dependency-aware experiment DAG state and scheduling decisions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum


class DagError(ValueError):
    pass


class NodeState(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    INTERRUPTED = "INTERRUPTED"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


class ResourceClass(StrEnum):
    CPU = "CPU"
    NETWORK = "NETWORK"
    GPU_HEAVY = "GPU_HEAVY"


@dataclass(frozen=True)
class ExperimentNode:
    experiment_id: str
    dependencies: tuple[str, ...] = ()
    resource: ResourceClass = ResourceClass.CPU
    state: NodeState = NodeState.PENDING
    protocol_id: str = ""
    run_id: str | None = None
    has_compatible_checkpoint: bool = False
    blocker: str | None = None

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["resource"] = self.resource.value
        value["state"] = self.state.value
        value["dependencies"] = list(self.dependencies)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ExperimentNode:
        raw_dependencies = value.get("dependencies", ())
        if not isinstance(raw_dependencies, (list, tuple)):
            raise DagError("node dependencies must be a list or tuple")
        return cls(
            experiment_id=str(value["experiment_id"]),
            dependencies=tuple(str(item) for item in raw_dependencies),
            resource=ResourceClass(str(value.get("resource", "CPU"))),
            state=NodeState(str(value.get("state", "PENDING"))),
            protocol_id=str(value.get("protocol_id", "")),
            run_id=str(value["run_id"]) if value.get("run_id") is not None else None,
            has_compatible_checkpoint=bool(value.get("has_compatible_checkpoint", False)),
            blocker=str(value["blocker"]) if value.get("blocker") is not None else None,
        )


@dataclass(frozen=True)
class ScheduleDecision:
    experiment_id: str
    action: str
    resource: ResourceClass
    reason: str


_TRANSITIONS: dict[NodeState, frozenset[NodeState]] = {
    NodeState.PENDING: frozenset({NodeState.RUNNING, NodeState.BLOCKED, NodeState.FAILED}),
    NodeState.RUNNING: frozenset(
        {NodeState.COMPLETE, NodeState.FAILED, NodeState.INTERRUPTED, NodeState.BLOCKED}
    ),
    NodeState.INTERRUPTED: frozenset({NodeState.RUNNING, NodeState.FAILED, NodeState.BLOCKED}),
    NodeState.FAILED: frozenset({NodeState.RUNNING, NodeState.BLOCKED}),
    NodeState.BLOCKED: frozenset({NodeState.PENDING, NodeState.RUNNING}),
    NodeState.COMPLETE: frozenset(),
}


@dataclass
class ExperimentDag:
    """Validated DAG whose terminal states survive orchestrator restarts."""

    nodes: dict[str, ExperimentNode] = field(default_factory=dict)

    @classmethod
    def build(cls, nodes: Iterable[ExperimentNode]) -> ExperimentDag:
        indexed: dict[str, ExperimentNode] = {}
        for node in nodes:
            if not node.experiment_id:
                raise DagError("experiment_id cannot be empty")
            if node.experiment_id in indexed:
                raise DagError(f"duplicate experiment ID: {node.experiment_id}")
            indexed[node.experiment_id] = node
        dag = cls(indexed)
        dag.validate()
        return dag

    def validate(self) -> None:
        active_gpu = [
            node.experiment_id
            for node in self.nodes.values()
            if node.state is NodeState.RUNNING and node.resource is ResourceClass.GPU_HEAVY
        ]
        if len(active_gpu) > 1:
            raise DagError(f"multiple heavy GPU owners are recorded: {sorted(active_gpu)}")
        for experiment_id, node in self.nodes.items():
            missing = sorted(set(node.dependencies) - set(self.nodes))
            if missing:
                raise DagError(f"{experiment_id} has unknown dependencies: {missing}")
            if experiment_id in node.dependencies:
                raise DagError(f"{experiment_id} depends on itself")
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(experiment_id: str) -> None:
            if experiment_id in visiting:
                raise DagError(f"dependency cycle includes {experiment_id}")
            if experiment_id in visited:
                return
            visiting.add(experiment_id)
            for dependency in self.nodes[experiment_id].dependencies:
                visit(dependency)
            visiting.remove(experiment_id)
            visited.add(experiment_id)

        for experiment_id in self.nodes:
            visit(experiment_id)

    def transition(
        self,
        experiment_id: str,
        target: NodeState,
        *,
        blocker: str | None = None,
        has_compatible_checkpoint: bool | None = None,
    ) -> None:
        node = self.nodes[experiment_id]
        if target in {NodeState.RUNNING, NodeState.COMPLETE} and not self.dependencies_complete(
            experiment_id
        ):
            raise DagError(f"cannot {target.value.lower()} {experiment_id} before its dependencies")
        if target is NodeState.RUNNING and node.resource is ResourceClass.GPU_HEAVY:
            competing = [
                item.experiment_id
                for item in self.nodes.values()
                if item.experiment_id != experiment_id
                and item.state is NodeState.RUNNING
                and item.resource is ResourceClass.GPU_HEAVY
            ]
            if competing:
                raise DagError(f"heavy GPU already owned by {sorted(competing)}")
        if target not in _TRANSITIONS[node.state]:
            raise DagError(f"illegal state transition: {node.state.value} -> {target.value}")
        if target is NodeState.BLOCKED and not blocker:
            raise DagError("a blocked node requires a blocker reason")
        values = node.to_dict()
        values["state"] = target.value
        values["blocker"] = blocker
        if has_compatible_checkpoint is not None:
            values["has_compatible_checkpoint"] = has_compatible_checkpoint
        self.nodes[experiment_id] = ExperimentNode.from_dict(values)

    def dependencies_complete(self, experiment_id: str) -> bool:
        return all(
            self.nodes[item].state is NodeState.COMPLETE
            for item in self.nodes[experiment_id].dependencies
        )

    def runnable(self, *, gpu_busy: bool = False) -> list[ScheduleDecision]:
        """Return runnable work without mutating DAG state.

        At most one heavy-GPU node is returned. CPU/network work remains eligible
        even when unrelated branches have failed or are blocked.
        """

        decisions: list[ScheduleDecision] = []
        gpu_selected = gpu_busy or any(
            node.state is NodeState.RUNNING and node.resource is ResourceClass.GPU_HEAVY
            for node in self.nodes.values()
        )
        for experiment_id in sorted(self.nodes):
            node = self.nodes[experiment_id]
            if node.state not in {NodeState.PENDING, NodeState.INTERRUPTED, NodeState.FAILED}:
                continue
            if not self.dependencies_complete(experiment_id):
                continue
            if node.resource is ResourceClass.GPU_HEAVY:
                if gpu_selected:
                    continue
                gpu_selected = True
            if node.state is NodeState.INTERRUPTED and node.has_compatible_checkpoint:
                action = "RESUME"
                reason = "interrupted node has a verified compatible checkpoint"
            elif node.state is NodeState.INTERRUPTED:
                action = "RESTART_RUN"
                reason = "interrupted node has no compatible complete checkpoint"
            elif node.state is NodeState.FAILED:
                action = "RECOVER"
                reason = "failed node is eligible for an authorized recovery attempt"
            else:
                action = "START"
                reason = "all dependencies are complete"
            decisions.append(ScheduleDecision(experiment_id, action, node.resource, reason))
        return decisions

    def propagate_dependency_blocks(self) -> list[str]:
        """Block only descendants that cannot run because a dependency is terminally blocked."""

        changed: list[str] = []
        made_progress = True
        while made_progress:
            made_progress = False
            for experiment_id, node in list(self.nodes.items()):
                if node.state is not NodeState.PENDING:
                    continue
                blockers = [
                    dependency
                    for dependency in node.dependencies
                    if self.nodes[dependency].state is NodeState.BLOCKED
                ]
                if blockers:
                    self.transition(
                        experiment_id,
                        NodeState.BLOCKED,
                        blocker=f"blocked dependencies: {', '.join(sorted(blockers))}",
                    )
                    changed.append(experiment_id)
                    made_progress = True
        return changed

    def snapshot(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "nodes": [self.nodes[item].to_dict() for item in sorted(self.nodes)],
        }

    @classmethod
    def from_snapshot(cls, value: Mapping[str, object]) -> ExperimentDag:
        raw_nodes = value.get("nodes")
        if not isinstance(raw_nodes, list):
            raise DagError("DAG snapshot must contain a node list")
        if any(not isinstance(item, Mapping) for item in raw_nodes):
            raise DagError("every DAG snapshot node must be a mapping")
        return cls.build(
            ExperimentNode.from_dict(item) for item in raw_nodes if isinstance(item, Mapping)
        )
