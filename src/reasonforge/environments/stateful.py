"""Purely local stateful tool sandbox used by synthetic agent tasks.

No method performs host filesystem, network, email, calendar, or payment
operations.  Every domain is represented by an in-memory state object that is
deep-copied for each rollout member.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import operator
import random
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any


class AuthorizationError(PermissionError):
    """Raised internally when a sensitive mutation was not authorized."""


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    value: Any = None
    error_code: str | None = None
    message: str | None = None


@dataclass(frozen=True)
class Transition:
    index: int
    call: ToolCall
    result: ToolResult
    state_before: str
    state_after: str


@dataclass
class SandboxState:
    scenario_id: str
    seed: int
    calendar: dict[str, dict[str, Any]] = field(default_factory=dict)
    reminders: dict[str, dict[str, Any]] = field(default_factory=dict)
    contacts: dict[str, dict[str, str]] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)
    retrieval_documents: dict[str, str] = field(default_factory=dict)
    sent_email: list[dict[str, str]] = field(default_factory=list)
    payment_intents: dict[str, dict[str, str]] = field(default_factory=dict)
    approved_actions: set[str] = field(default_factory=set)

    def clone(self) -> SandboxState:
        return copy.deepcopy(self)

    def canonical_json(self) -> str:
        value = asdict(self)
        value["approved_actions"] = sorted(self.approved_actions)
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)

    @property
    def identity(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()


def _required(args: Mapping[str, Any], *names: str) -> tuple[Any, ...]:
    missing = [name for name in names if name not in args or args[name] in (None, "")]
    if missing:
        raise ValueError("missing required argument(s): " + ", ".join(missing))
    return tuple(args[name] for name in names)


class StatefulToolEnvironment:
    """Execute typed calls against a private in-memory sandbox."""

    DEFAULT_PROTECTED_ACTIONS = frozenset({"email.send", "payment_intent.create", "contacts.add"})
    TOOL_ARGUMENTS: Mapping[str, tuple[str, ...]] = {
        "assistant.request_clarification": ("question",),
        "calendar.create_event": ("title", "start", "end"),
        "calendar.list_events": (),
        "reminders.create": ("text", "due"),
        "reminders.complete": ("reminder_id",),
        "contacts.add": ("name", "email"),
        "contacts.find": ("query",),
        "calculator.evaluate": ("expression",),
        "notes.write": ("title", "content"),
        "notes.read": ("title",),
        "files.write": ("path", "content"),
        "files.read": ("path",),
        "retrieval.search": ("query",),
        "email.send": ("recipient", "subject", "body"),
        "payment_intent.create": ("recipient", "amount", "currency"),
    }

    def __init__(
        self,
        initial_state: SandboxState,
        *,
        protected_actions: Iterable[str] = DEFAULT_PROTECTED_ACTIONS,
    ) -> None:
        self._initial_state = initial_state.clone()
        self.state = initial_state.clone()
        self.protected_actions = frozenset(protected_actions)
        self.trace: list[Transition] = []

    @property
    def initial_state_identity(self) -> str:
        return self._initial_state.identity

    def reset(self) -> None:
        self.state = self._initial_state.clone()
        self.trace.clear()

    def tool_schemas(self) -> tuple[Mapping[str, Any], ...]:
        """Return deterministic JSON-serializable function schemas."""

        return tuple(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            argument: {
                                "type": (
                                    "number"
                                    if name == "payment_intent.create" and argument == "amount"
                                    else "string"
                                )
                            }
                            for argument in arguments
                        },
                        "required": list(arguments),
                        "additionalProperties": False,
                    },
                },
            }
            for name, arguments in sorted(self.TOOL_ARGUMENTS.items())
        )

    def authorize(self, *actions: str) -> None:
        unknown = set(actions) - self.protected_actions
        if unknown:
            raise ValueError(f"actions are not authorization-gated: {sorted(unknown)}")
        self.state.approved_actions.update(actions)

    def call(self, name: str, arguments: Mapping[str, Any] | None = None) -> ToolResult:
        args = dict(arguments or {})
        call = ToolCall(name=name, arguments=args)
        before = self.state.identity
        try:
            if name in self.protected_actions and name not in self.state.approved_actions:
                raise AuthorizationError(f"explicit authorization required for {name}")
            handler_name = "_tool_" + name.replace(".", "_")
            handler = getattr(self, handler_name, None)
            if handler is None:
                result = ToolResult(
                    False, error_code="UNKNOWN_TOOL", message=f"unknown tool: {name}"
                )
            else:
                value = handler(args)
                result = ToolResult(True, value=value)
        except AuthorizationError as exc:
            result = ToolResult(False, error_code="AUTHORIZATION_REQUIRED", message=str(exc))
        except ValueError as exc:
            result = ToolResult(False, error_code="INVALID_ARGUMENT", message=str(exc))
        except (KeyError, FileNotFoundError) as exc:
            result = ToolResult(False, error_code="NOT_FOUND", message=str(exc))
        after = self.state.identity
        self.trace.append(Transition(len(self.trace), call, result, before, after))
        return result

    def _next_id(self, prefix: str, collection: Mapping[str, Any]) -> str:
        return f"{prefix}-{len(collection) + 1:04d}"

    def _tool_calendar_create_event(self, args: Mapping[str, Any]) -> Mapping[str, Any]:
        title, start, end = _required(args, "title", "start", "end")
        if str(end) <= str(start):
            raise ValueError("end must be after start")
        event_id = self._next_id("event", self.state.calendar)
        event = {"id": event_id, "title": str(title), "start": str(start), "end": str(end)}
        self.state.calendar[event_id] = event
        return event

    def _tool_calendar_list_events(self, args: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        del args
        return [self.state.calendar[key].copy() for key in sorted(self.state.calendar)]

    def _tool_reminders_create(self, args: Mapping[str, Any]) -> Mapping[str, Any]:
        text, due = _required(args, "text", "due")
        reminder_id = self._next_id("reminder", self.state.reminders)
        reminder = {"id": reminder_id, "text": str(text), "due": str(due), "done": False}
        self.state.reminders[reminder_id] = reminder
        return reminder

    def _tool_reminders_complete(self, args: Mapping[str, Any]) -> Mapping[str, Any]:
        (reminder_id,) = _required(args, "reminder_id")
        reminder = self.state.reminders[str(reminder_id)]
        reminder["done"] = True
        return reminder.copy()

    def _tool_contacts_add(self, args: Mapping[str, Any]) -> Mapping[str, str]:
        name, email = _required(args, "name", "email")
        if "@" not in str(email):
            raise ValueError("email must contain @")
        contact_id = self._next_id("contact", self.state.contacts)
        contact = {"id": contact_id, "name": str(name), "email": str(email)}
        self.state.contacts[contact_id] = contact
        return contact

    def _tool_contacts_find(self, args: Mapping[str, Any]) -> list[Mapping[str, str]]:
        (query,) = _required(args, "query")
        needle = str(query).casefold()
        return [
            contact.copy()
            for _, contact in sorted(self.state.contacts.items())
            if needle in contact["name"].casefold() or needle in contact["email"].casefold()
        ]

    def _tool_calculator_evaluate(self, args: Mapping[str, Any]) -> str:
        (expression,) = _required(args, "expression")
        result = _safe_calculate(str(expression))
        return format(result, "f")

    def _tool_notes_write(self, args: Mapping[str, Any]) -> Mapping[str, str]:
        title, content = _required(args, "title", "content")
        self.state.notes[str(title)] = str(content)
        return {"title": str(title), "content": str(content)}

    def _tool_notes_read(self, args: Mapping[str, Any]) -> str:
        (title,) = _required(args, "title")
        return self.state.notes[str(title)]

    def _tool_files_write(self, args: Mapping[str, Any]) -> Mapping[str, str]:
        path, content = _required(args, "path", "content")
        safe_path = _logical_path(str(path))
        self.state.files[safe_path] = str(content)
        return {"path": safe_path, "content": str(content)}

    def _tool_files_read(self, args: Mapping[str, Any]) -> str:
        (path,) = _required(args, "path")
        return self.state.files[_logical_path(str(path))]

    def _tool_retrieval_search(self, args: Mapping[str, Any]) -> list[Mapping[str, str]]:
        (query,) = _required(args, "query")
        terms = {term.casefold() for term in str(query).split() if term}
        scored: list[tuple[int, str, str]] = []
        for doc_id, text in self.state.retrieval_documents.items():
            score = sum(term in text.casefold() for term in terms)
            if score:
                scored.append((score, doc_id, text))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [{"document_id": doc_id, "text": text} for _, doc_id, text in scored]

    def _tool_email_send(self, args: Mapping[str, Any]) -> Mapping[str, str]:
        recipient, subject, body = _required(args, "recipient", "subject", "body")
        if "@" not in str(recipient):
            raise ValueError("recipient must be an email address")
        message = {"recipient": str(recipient), "subject": str(subject), "body": str(body)}
        self.state.sent_email.append(message)
        return message.copy()

    def _tool_payment_intent_create(self, args: Mapping[str, Any]) -> Mapping[str, str]:
        recipient, amount, currency = _required(args, "recipient", "amount", "currency")
        try:
            decimal_amount = Decimal(str(amount))
        except InvalidOperation as exc:
            raise ValueError("amount must be numeric") from exc
        if decimal_amount <= 0:
            raise ValueError("amount must be positive")
        intent_id = self._next_id("payment", self.state.payment_intents)
        intent = {
            "id": intent_id,
            "recipient": str(recipient),
            "amount": format(decimal_amount, "f"),
            "currency": str(currency).upper(),
            "status": "intent_only",
        }
        self.state.payment_intents[intent_id] = intent
        return intent.copy()

    def _tool_assistant_request_clarification(self, args: Mapping[str, Any]) -> Mapping[str, str]:
        (question,) = _required(args, "question")
        return {"question": str(question)}


def _logical_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    if normalized.startswith("/") or ":" in normalized or ".." in normalized.split("/"):
        raise ValueError("path must remain inside the mock filesystem")
    cleaned = "/".join(part for part in normalized.split("/") if part not in ("", "."))
    if not cleaned:
        raise ValueError("path must not be empty")
    return cleaned


_BINARY_OPERATORS: Mapping[type[ast.operator], Callable[[Decimal, Decimal], Decimal]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARY_OPERATORS: Mapping[type[ast.unaryop], Callable[[Decimal], Decimal]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_calculate(expression: str) -> Decimal:
    """Evaluate arithmetic without names, calls, attributes, or host access."""

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("invalid arithmetic expression") from exc

    def evaluate(node: ast.AST) -> Decimal:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return Decimal(str(node.value))
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            try:
                return _BINARY_OPERATORS[type(node.op)](evaluate(node.left), evaluate(node.right))
            except (ArithmeticError, InvalidOperation) as exc:
                raise ValueError("invalid arithmetic operation") from exc
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            return _UNARY_OPERATORS[type(node.op)](evaluate(node.operand))
        raise ValueError("only numeric literals and +, -, *, / are allowed")

    return evaluate(tree)


class GroupedEnvironmentFactory:
    """Create isolated environments with deterministic GRPO group state.

    A group key deterministically identifies one frozen scenario.  Every member
    receives a fresh deep copy of that scenario, so mutations cannot leak across
    rollouts.  Distinct group keys receive distinct scenario identities and RNG
    streams even when their human-visible fixture contents happen to match.
    """

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def deterministic_group_key(self, task_id: str, optimizer_step: int) -> str:
        if optimizer_step < 0:
            raise ValueError("optimizer_step must be non-negative")
        payload = f"rf-group-v1:{self.seed}:{optimizer_step}:{task_id}"
        return hashlib.sha256(payload.encode()).hexdigest()[:24]

    def create(self, group_key: str) -> StatefulToolEnvironment:
        if not group_key:
            raise ValueError("group_key must not be empty")
        digest = hashlib.sha256(f"rf-state-v1:{self.seed}:{group_key}".encode()).digest()
        scenario_seed = int.from_bytes(digest[:8], "big")
        rng = random.Random(scenario_seed)
        contact_name = ("Ari", "Bo", "Cleo", "Devi")[rng.randrange(4)]
        state = SandboxState(
            scenario_id=f"scenario-{group_key}",
            seed=scenario_seed,
            contacts={
                "contact-0001": {
                    "id": "contact-0001",
                    "name": contact_name,
                    "email": f"{contact_name.casefold()}@example.invalid",
                }
            },
            retrieval_documents={
                "policy": "Payment intents require explicit user authorization.",
                "hours": f"The help desk closes at {16 + rng.randrange(3)}:00.",
            },
        )
        return StatefulToolEnvironment(state)
