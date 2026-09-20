"""Journaled v2 writer for recovery of v1 protected output sets.

This module is additive.  It never rewrites an existing valid record, shard, or
manifest during adoption.  New appends use a write-ahead journal so interruption
between shard fsync and manifest publication can be recovered deterministically.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from reasonforge.checkpoints.hashing import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    sha256_file,
)
from reasonforge.checkpoints.models import utc_now

from .protected_outputs import (
    ProtectedOutputIntegrityError,
    ProtectedOutputManifest,
    ProtectedOutputRecord,
    ProtectedOutputShard,
    ProtectedOutputVerification,
    ProtectedOutputWriter,
)

WRITER_PROTOCOL_ID = "RFPT-PROTECTED-WRITER-v2"
_APPEND_JOURNAL = "WRITER_V2_PENDING_APPEND.json"
_COMPLETE_JOURNAL = "WRITER_V2_PENDING_COMPLETE.json"
_ADOPTION = "WRITER_V2_ADOPTION.json"
_MANIFEST = "manifest.json"
_COMPLETE = "PROTECTED_OUTPUT_COMPLETE.json"


class JournaledProtectedOutputWriter:
    """Crash-recoverable facade over the immutable v1 output schema."""

    def __init__(self, output_directory: Path | str) -> None:
        self.output_directory = Path(output_directory)
        self.base = ProtectedOutputWriter(self.output_directory)

    @classmethod
    def adopt(
        cls, output_directory: Path | str, *, run_plan_sha256: str
    ) -> JournaledProtectedOutputWriter:
        writer = cls(output_directory)
        writer.base.verify()
        sidecar = writer.output_directory / _ADOPTION
        expected = {
            "schema_version": 1,
            "status": "ADOPTED",
            "writer_protocol_id": WRITER_PROTOCOL_ID,
            "run_plan_sha256": run_plan_sha256,
            "adopted_manifest_sha256": sha256_file(writer.output_directory / _MANIFEST),
            "finish_reason_semantics": "LEGACY_FIELD_UNVERIFIED_DO_NOT_USE_FOR_CLAIMS",
        }
        if sidecar.exists():
            observed = json.loads(sidecar.read_text("utf-8"))
            if (
                not isinstance(observed, dict)
                or observed.get("writer_protocol_id") != WRITER_PROTOCOL_ID
                or observed.get("run_plan_sha256") != run_plan_sha256
                or observed.get("finish_reason_semantics")
                != "LEGACY_FIELD_UNVERIFIED_DO_NOT_USE_FOR_CLAIMS"
            ):
                raise ProtectedOutputIntegrityError("v2 adoption identity mismatch")
        else:
            atomic_write_json(sidecar, expected)
        return writer

    @classmethod
    def open(cls, output_directory: Path | str) -> JournaledProtectedOutputWriter:
        writer = cls(output_directory)
        adoption_path = writer.output_directory / _ADOPTION
        if not adoption_path.is_file():
            raise ProtectedOutputIntegrityError("output set has not been adopted by writer v2")
        adoption = json.loads(adoption_path.read_text("utf-8"))
        run_plan_path = writer.output_directory / "RUN_PLAN.json"
        if (
            not isinstance(adoption, dict)
            or adoption.get("writer_protocol_id") != WRITER_PROTOCOL_ID
            or not run_plan_path.is_file()
            or adoption.get("run_plan_sha256") != sha256_file(run_plan_path)
        ):
            raise ProtectedOutputIntegrityError("v2 adoption does not bind the current run plan")
        writer.recover()
        writer.base.verify()
        return writer

    def verify(self, *, require_complete: bool = False) -> ProtectedOutputVerification:
        self.recover()
        return self.base.verify(require_complete=require_complete)

    def append(self, record: ProtectedOutputRecord) -> None:
        verification = self.verify()
        manifest = verification.manifest
        if manifest.status == "COMPLETE":
            raise ProtectedOutputIntegrityError("cannot append to a complete output set")
        record.validate()
        ProtectedOutputWriter._assert_record_identity(record, manifest.identity)
        if record.example_id not in manifest.expected_example_ids:
            raise ProtectedOutputIntegrityError(f"unexpected example ID: {record.example_id}")
        if record.example_id in verification.record_sha256:
            raise ProtectedOutputIntegrityError(f"duplicate example ID: {record.example_id}")
        shards = list(manifest.shards)
        if shards and shards[-1].status == "IN_PROGRESS":
            shard_index = len(shards) - 1
            prior_shard = shards[-1]
        else:
            shard_index = len(shards)
            prior_shard = None
        filename = f"shard_{shard_index:03d}.jsonl"
        shard_path = self.output_directory / filename
        prior_bytes = shard_path.read_bytes() if shard_path.exists() else b""
        line = canonical_json_bytes(record.to_dict())
        journal = {
            "schema_version": 1,
            "operation": "APPEND",
            "writer_protocol_id": WRITER_PROTOCOL_ID,
            "prior_manifest_sha256": sha256_file(self.output_directory / _MANIFEST),
            "filename": filename,
            "prior_byte_size": len(prior_bytes),
            "prior_prefix_sha256": hashlib.sha256(prior_bytes).hexdigest(),
            "record": record.to_dict(),
            "record_line_sha256": hashlib.sha256(line).hexdigest(),
        }
        atomic_write_json(self.output_directory / _APPEND_JOURNAL, journal)
        with shard_path.open("ab") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        records, _ = ProtectedOutputWriter._read_shard(shard_path)
        updated = ProtectedOutputShard(
            filename=filename,
            status=("COMPLETE" if len(records) >= manifest.shard_max_records else "IN_PROGRESS"),
            sha256=sha256_file(shard_path),
            byte_size=shard_path.stat().st_size,
            record_count=len(records),
            example_ids=[item.example_id for item in records],
        )
        if prior_shard is None:
            shards.append(updated)
        else:
            shards[-1] = updated
        atomic_write_json(
            self.output_directory / _MANIFEST, replace(manifest, shards=shards).to_dict()
        )
        (self.output_directory / _APPEND_JOURNAL).unlink()

    def complete(self) -> ProtectedOutputVerification:
        verification = self.verify()
        if verification.manifest.status == "COMPLETE":
            return verification
        if verification.missing_example_ids:
            raise ProtectedOutputIntegrityError("cannot complete an output set with missing IDs")
        completed_at = utc_now()
        atomic_write_json(
            self.output_directory / _COMPLETE_JOURNAL,
            {
                "schema_version": 1,
                "operation": "COMPLETE",
                "writer_protocol_id": WRITER_PROTOCOL_ID,
                "prior_manifest_sha256": sha256_file(self.output_directory / _MANIFEST),
                "completed_at": completed_at,
            },
        )
        self._finish_completion(verification.manifest, completed_at)
        (self.output_directory / _COMPLETE_JOURNAL).unlink()
        return self.base.verify(require_complete=True)

    def recover(self) -> None:
        append_journal = self.output_directory / _APPEND_JOURNAL
        complete_journal = self.output_directory / _COMPLETE_JOURNAL
        if append_journal.exists() and complete_journal.exists():
            raise ProtectedOutputIntegrityError("conflicting v2 recovery journals")
        if append_journal.exists():
            self._recover_append(self._read_journal(append_journal, "APPEND"))
            append_journal.unlink()
        if complete_journal.exists():
            journal = self._read_journal(complete_journal, "COMPLETE")
            manifest = self._load_manifest()
            if manifest.status == "IN_PROGRESS":
                verification = self.base.verify()
                if verification.missing_example_ids:
                    raise ProtectedOutputIntegrityError("completion journal has missing IDs")
                self._finish_completion(manifest, str(journal["completed_at"]))
            elif not (self.output_directory / _COMPLETE).exists():
                self._validate_complete_manifest_without_sentinel(manifest)
                self._write_complete_sentinel(manifest)
            self.base.verify(require_complete=True)
            complete_journal.unlink()

    def _recover_append(self, journal: dict[str, Any]) -> None:
        record = ProtectedOutputRecord.from_dict(journal["record"])
        try:
            verification = self.base.verify()
        except ProtectedOutputIntegrityError:
            verification = None
        if verification is not None and record.example_id in verification.record_sha256:
            return
        manifest = self._load_manifest()
        if sha256_file(self.output_directory / _MANIFEST) != journal["prior_manifest_sha256"]:
            raise ProtectedOutputIntegrityError("append recovery manifest ancestry mismatch")
        shard_path = self.output_directory / str(journal["filename"])
        current = shard_path.read_bytes() if shard_path.exists() else b""
        prior_size = int(journal["prior_byte_size"])
        if len(current) < prior_size:
            raise ProtectedOutputIntegrityError("append recovery shard is shorter than prior state")
        prefix = current[:prior_size]
        if hashlib.sha256(prefix).hexdigest() != journal["prior_prefix_sha256"]:
            raise ProtectedOutputIntegrityError("append recovery shard prefix mismatch")
        line = canonical_json_bytes(record.to_dict())
        tail = current[prior_size:]
        if tail != line:
            if not line.startswith(tail):
                raise ProtectedOutputIntegrityError("append recovery has an ambiguous shard tail")
            atomic_write_bytes(shard_path, prefix + line)
        records, _ = ProtectedOutputWriter._read_shard(shard_path)
        shards = list(manifest.shards)
        updated = ProtectedOutputShard(
            filename=shard_path.name,
            status=("COMPLETE" if len(records) >= manifest.shard_max_records else "IN_PROGRESS"),
            sha256=sha256_file(shard_path),
            byte_size=shard_path.stat().st_size,
            record_count=len(records),
            example_ids=[item.example_id for item in records],
        )
        if shards and shards[-1].filename == shard_path.name:
            shards[-1] = updated
        elif shard_path.name == f"shard_{len(shards):03d}.jsonl":
            shards.append(updated)
        else:
            raise ProtectedOutputIntegrityError("append recovery shard position mismatch")
        atomic_write_json(
            self.output_directory / _MANIFEST, replace(manifest, shards=shards).to_dict()
        )

    def _finish_completion(self, manifest: ProtectedOutputManifest, completed_at: str) -> None:
        complete_manifest = replace(
            manifest,
            shards=[replace(shard, status="COMPLETE") for shard in manifest.shards],
            status="COMPLETE",
            completed_at=completed_at,
        )
        atomic_write_json(self.output_directory / _MANIFEST, complete_manifest.to_dict())
        self._write_complete_sentinel(complete_manifest)

    def _write_complete_sentinel(self, manifest: ProtectedOutputManifest) -> None:
        atomic_write_json(
            self.output_directory / _COMPLETE,
            {
                "status": "COMPLETE",
                "output_set_id": manifest.identity.output_set_id,
                "manifest_sha256": sha256_file(self.output_directory / _MANIFEST),
                "completed_at": manifest.completed_at,
            },
        )

    def _validate_complete_manifest_without_sentinel(
        self, manifest: ProtectedOutputManifest
    ) -> None:
        if manifest.status != "COMPLETE" or not manifest.completed_at:
            raise ProtectedOutputIntegrityError("completion recovery manifest is not complete")
        observed: set[str] = set()
        for shard in manifest.shards:
            path = self.output_directory / shard.filename
            if sha256_file(path) != shard.sha256 or path.stat().st_size != shard.byte_size:
                raise ProtectedOutputIntegrityError("completion recovery shard mismatch")
            records, _ = ProtectedOutputWriter._read_shard(path)
            if [item.example_id for item in records] != shard.example_ids:
                raise ProtectedOutputIntegrityError("completion recovery inventory mismatch")
            observed.update(item.example_id for item in records)
        if observed != set(manifest.expected_example_ids):
            raise ProtectedOutputIntegrityError("completion recovery example inventory mismatch")

    def _load_manifest(self) -> ProtectedOutputManifest:
        value = json.loads((self.output_directory / _MANIFEST).read_text("utf-8"))
        return ProtectedOutputManifest.from_dict(value)

    @staticmethod
    def _read_journal(path: Path, operation: str) -> dict[str, Any]:
        value = json.loads(path.read_text("utf-8"))
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != 1
            or value.get("operation") != operation
            or value.get("writer_protocol_id") != WRITER_PROTOCOL_ID
        ):
            raise ProtectedOutputIntegrityError(f"invalid {operation.lower()} recovery journal")
        return value
