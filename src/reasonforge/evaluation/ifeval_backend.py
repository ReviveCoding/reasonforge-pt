"""Fail-closed qualification of the pinned official IFEval scorer backend."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from reasonforge.checkpoints.hashing import sha256_file

from .protected_outputs import ProtectedOutputIntegrityError
from .protected_protocol import directory_tree_sha256


def _git_blob_sha1(path: Path) -> str:
    # The pinned checkout uses Windows CRLF worktree conversion; Git object IDs
    # are computed over the canonical LF blob bytes.
    payload = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha1(b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload).hexdigest()


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtectedOutputIntegrityError(f"IFEval backend {label} must be a mapping")
    return value


def qualify_ifeval_backend(config_path: Path) -> dict[str, Any]:
    """Validate and activate only the frozen official evaluator and tokenizer data."""

    try:
        raw = yaml.safe_load(config_path.read_text("utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ProtectedOutputIntegrityError(f"invalid IFEval backend config: {error}") from error
    config = _mapping(raw, "config")
    if (
        config.get("schema_version") != 1
        or config.get("backend_id") != "RFPT-IFEVAL-BACKEND-v1"
        or config.get("google_research_commit") != "5594ac0ee7a13c77eb1e0b98f0f305a2ca65b6c4"
        or config.get("nltk_data_commit") != "550b6625bcef1f2abff2ff770a5a0d272c9c6b2a"
        or config.get("langdetect_seed") != 0
        or config.get("semantic_outputs_accessed") is not False
        or config.get("scores_computed") is not False
    ):
        raise ProtectedOutputIntegrityError("IFEval backend scientific identity is invalid")

    source_root = Path(str(config.get("source_root"))).resolve()
    source_files = _mapping(config.get("source_files"), "source files")
    source_hashes: dict[str, str] = {}
    for relative, identity_raw in source_files.items():
        identity = _mapping(identity_raw, f"source identity {relative}")
        candidate = source_root / str(relative)
        expected_sha256 = identity.get("sha256")
        expected_blob = identity.get("git_blob")
        if (
            not candidate.is_file()
            or not isinstance(expected_sha256, str)
            or sha256_file(candidate) != expected_sha256
            or not isinstance(expected_blob, str)
            or _git_blob_sha1(candidate) != expected_blob
        ):
            raise ProtectedOutputIntegrityError(f"IFEval source identity mismatch: {relative}")
        source_hashes[str(relative)] = expected_sha256

    requirements_lock = Path(str(config.get("requirements_lock")))
    if not requirements_lock.is_file():
        raise ProtectedOutputIntegrityError("IFEval dependency lock is missing")
    packages = _mapping(config.get("packages"), "packages")
    installed: dict[str, str] = {}
    for package, expected_version in packages.items():
        try:
            observed = importlib.metadata.version(str(package))
        except importlib.metadata.PackageNotFoundError as error:
            raise ProtectedOutputIntegrityError(
                f"IFEval dependency is missing: {package}"
            ) from error
        if observed != str(expected_version):
            raise ProtectedOutputIntegrityError(
                f"IFEval dependency version mismatch: {package}={observed}"
            )
        installed[str(package)] = observed

    archives = _mapping(config.get("nltk_archives"), "NLTK archives")
    for raw_path, expected_sha256 in archives.items():
        archive = Path(str(raw_path))
        if (
            not archive.is_file()
            or not isinstance(expected_sha256, str)
            or sha256_file(archive) != expected_sha256
        ):
            raise ProtectedOutputIntegrityError(f"NLTK archive identity mismatch: {archive}")
    nltk_root = Path(str(config.get("nltk_data_root"))).resolve()
    trees = _mapping(config.get("nltk_extracted_trees"), "NLTK extracted trees")
    for relative, expected_sha256 in trees.items():
        tree = nltk_root / str(relative)
        if (
            not tree.is_dir()
            or not isinstance(expected_sha256, str)
            or directory_tree_sha256(tree) != expected_sha256
        ):
            raise ProtectedOutputIntegrityError(f"NLTK extracted-tree mismatch: {relative}")

    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    import nltk  # type: ignore[import-untyped]
    from langdetect import DetectorFactory  # type: ignore[import-untyped]

    nltk_text = str(nltk_root)
    if nltk_text not in nltk.data.path:
        nltk.data.path.insert(0, nltk_text)
    DetectorFactory.seed = 0
    try:
        registry_module = importlib.import_module(
            "instruction_following_eval.instructions_registry"
        )
        registry = registry_module.INSTRUCTION_DICT
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("tokenizers/punkt_tab")
    except (ImportError, AttributeError, LookupError) as error:
        raise ProtectedOutputIntegrityError(
            "pinned official IFEval runtime could not be activated"
        ) from error
    expected_registry = (
        source_root / "instruction_following_eval/instructions_registry.py"
    ).resolve()
    observed_registry = Path(str(registry_module.__file__)).resolve()
    if observed_registry != expected_registry:
        raise ProtectedOutputIntegrityError("IFEval imported from an unauthorized source path")
    expected_count = config.get("expected_instruction_count")
    if not isinstance(registry, Mapping) or len(registry) != expected_count:
        raise ProtectedOutputIntegrityError("IFEval instruction registry identity is invalid")
    return {
        "status": "QUALIFIED",
        "backend_id": config["backend_id"],
        "config_sha256": sha256_file(config_path),
        "source_commit": config["google_research_commit"],
        "source_hashes": dict(sorted(source_hashes.items())),
        "requirements_lock_sha256": sha256_file(requirements_lock),
        "packages": dict(sorted(installed.items())),
        "nltk_data_commit": config["nltk_data_commit"],
        "instruction_count": len(registry),
        "langdetect_seed": DetectorFactory.seed,
        "semantic_outputs_accessed": False,
        "scores_computed": False,
    }
