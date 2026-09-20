"""Process-safe ownership of the single qualified CUDA device."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class GpuOwnerLease:
    """Hold an OS-backed exclusive lease for the one physical CUDA GPU."""

    def __init__(self, workspace: Path, experiment_id: str, run_id: str) -> None:
        self.path = workspace / "artifacts" / "GPU_OWNER.lock"
        self.experiment_id = experiment_id
        self.run_id = run_id
        self._handle: Any = None

    def __enter__(self) -> GpuOwnerLease:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+b")
        self._handle.seek(0, os.SEEK_END)
        if self._handle.tell() == 0:
            self._handle.write(b"\0")
            self._handle.flush()
            os.fsync(self._handle.fileno())
        self._handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(  # type: ignore[attr-defined]
                    self._handle.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,  # type: ignore[attr-defined]
                )
        except OSError as error:
            self._handle.close()
            self._handle = None
            raise RuntimeError("the physical CUDA GPU already has a ReasonForge owner") from error
        self._handle.seek(0)
        self._handle.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "experiment_id": self.experiment_id,
                    "run_id": self.run_id,
                }
            ).encode("utf-8")
        )
        self._handle.truncate()
        self._handle.flush()
        os.fsync(self._handle.fileno())
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._handle is None:
            return
        self._handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(  # type: ignore[attr-defined]
                self._handle.fileno(),
                fcntl.LOCK_UN,  # type: ignore[attr-defined]
            )
        self._handle.close()
        self._handle = None
