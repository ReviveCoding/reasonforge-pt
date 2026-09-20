"""Reproducible host and CUDA inventory."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil

PACKAGES = (
    "torch",
    "transformers",
    "trl",
    "peft",
    "accelerate",
    "bitsandbytes",
    "datasets",
    "polars",
    "pyarrow",
    "duckdb",
)


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _nvidia_inventory() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"qualified": False, "error": "nvidia-smi not found"}
    query = (
        "index,name,uuid,driver_version,memory.total,memory.free,"
        "compute_cap,pstate,temperature.gpu,power.limit"
    )
    result = subprocess.run(
        [executable, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return {
        "qualified": result.returncode == 0,
        "query": query,
        "rows": [line.strip() for line in result.stdout.splitlines() if line.strip()],
        "stderr": result.stderr.strip(),
    }


def collect_environment(*, exercise_cuda: bool = True) -> dict[str, Any]:
    usage = shutil.disk_usage(Path.cwd())
    memory = psutil.virtual_memory()
    snapshot: dict[str, Any] = {
        "created_at": datetime.now().astimezone().isoformat(),
        "platform": platform.platform(),
        "windows_version": platform.win32_ver(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version,
        "executable": sys.executable,
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "ram_total_bytes": memory.total,
        "ram_available_bytes": memory.available,
        "disk_total_bytes": usage.total,
        "disk_free_bytes": usage.free,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "packages": _package_versions(),
        "nvidia_smi": _nvidia_inventory(),
        "wsl": {"qualified": False, "reason": "enumeration permission denied at bootstrap"},
    }
    try:
        import torch

        cuda: dict[str, Any] = {
            "available": torch.cuda.is_available(),
            "torch_cuda": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
        }
        if torch.cuda.is_available():
            cuda.update(
                {
                    "name": torch.cuda.get_device_name(0),
                    "capability": list(torch.cuda.get_device_capability(0)),
                    "bf16_supported": torch.cuda.is_bf16_supported(),
                }
            )
            if exercise_cuda:
                left = torch.randn((1024, 1024), device="cuda", dtype=torch.bfloat16)
                product = left @ left
                torch.cuda.synchronize()
                cuda["bf16_matmul_finite"] = bool(torch.isfinite(product).all().item())
                cuda["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                del left, product
                torch.cuda.empty_cache()
        snapshot["cuda"] = cuda
    # A doctor command must report arbitrary backend import/runtime errors instead of crashing.
    except Exception as error:  # pragma: no cover
        snapshot["cuda"] = {"available": False, "error": repr(error)}
    return snapshot


def write_environment(snapshot: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    json.loads(temporary.read_text(encoding="utf-8"))
    temporary.replace(path)
