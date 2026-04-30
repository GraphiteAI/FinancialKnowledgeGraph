"""
Subprocess-based sandbox for re-executing miner causal-inference code.

The validator calls `SandboxRunner.run(code, data)` with bytes fetched from
the ArtifactStore; the runner writes them to a temp dir, launches a fresh
Python process, and parses a JSON result printed to stdout.

Security envelope (v1):
- Hard wall-clock timeout (default 30s)
- Posix `resource` limits on CPU, address space, and file size (Linux/macOS)
- No inherited environment variables beyond PATH + PYTHONPATH
- Network is NOT blocked at the kernel level in v1 — miners who call out
  will still produce deterministic results only if the network call is
  deterministic. Phase 3 moves this to `unshare`/Docker/firejail.

The contract for miner code:
- Reads the input data file path from argv[1]
- Writes exactly one JSON object to stdout: {"effect_size": float, ...}
- Any extra stdout before the JSON is ignored; stderr is captured for debug
"""
from __future__ import annotations

import json
import logging
import os
import resource
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("sn43.sandbox")


class SandboxTimeoutError(RuntimeError):
    pass


@dataclass
class SandboxResult:
    ok: bool
    output: dict = field(default_factory=dict)
    stderr: str = ""
    stdout: str = ""
    error: Optional[str] = None
    duration_seconds: float = 0.0


class SandboxRunner:
    """Executes code bytes against data bytes in a hardened subprocess."""

    def __init__(
        self,
        timeout_seconds: float = 30.0,
        max_memory_bytes: int = 512 * 1024 * 1024,   # 512 MB
        max_cpu_seconds: int = 30,
        max_output_bytes: int = 4 * 1024 * 1024,     # 4 MB
        python_executable: Optional[str] = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_memory_bytes = max_memory_bytes
        self.max_cpu_seconds = max_cpu_seconds
        self.max_output_bytes = max_output_bytes
        self.python_executable = python_executable or sys.executable

    def run(self, code: bytes, data: bytes) -> SandboxResult:
        import time

        start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="sn43_sandbox_") as tmp_str:
            tmp = Path(tmp_str)
            code_path = tmp / "miner_code.py"
            data_path = tmp / "input_data.bin"
            code_path.write_bytes(code)
            data_path.write_bytes(data)

            env = {
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
                "LC_ALL": "C.UTF-8",
                "LANG": "C.UTF-8",
                "HOME": str(tmp),
                "TMPDIR": str(tmp),
            }

            try:
                proc = subprocess.Popen(
                    [self.python_executable, "-I", str(code_path), str(data_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.DEVNULL,
                    env=env,
                    cwd=str(tmp),
                    preexec_fn=self._apply_limits,
                )
            except Exception as e:  # noqa: BLE001
                return SandboxResult(
                    ok=False,
                    error=f"spawn failed: {e}",
                    duration_seconds=time.monotonic() - start,
                )

            try:
                stdout_bytes, stderr_bytes = proc.communicate(
                    timeout=self.timeout_seconds
                )
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                return SandboxResult(
                    ok=False,
                    error=f"timeout after {self.timeout_seconds}s",
                    duration_seconds=self.timeout_seconds,
                )

            duration = time.monotonic() - start
            stdout = stdout_bytes[: self.max_output_bytes].decode("utf-8", "replace")
            stderr = stderr_bytes[: self.max_output_bytes].decode("utf-8", "replace")

            if proc.returncode != 0:
                return SandboxResult(
                    ok=False,
                    stdout=stdout,
                    stderr=stderr,
                    error=f"exit {proc.returncode}",
                    duration_seconds=duration,
                )

            parsed = _parse_last_json_object(stdout)
            if parsed is None:
                return SandboxResult(
                    ok=False,
                    stdout=stdout,
                    stderr=stderr,
                    error="no JSON object found on stdout",
                    duration_seconds=duration,
                )

            return SandboxResult(
                ok=True,
                output=parsed,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=duration,
            )

    def _apply_limits(self) -> None:
        """Called in the child after fork, before exec."""
        try:
            resource.setrlimit(
                resource.RLIMIT_CPU,
                (self.max_cpu_seconds, self.max_cpu_seconds),
            )
            resource.setrlimit(
                resource.RLIMIT_AS,
                (self.max_memory_bytes, self.max_memory_bytes),
            )
            resource.setrlimit(
                resource.RLIMIT_FSIZE,
                (self.max_output_bytes, self.max_output_bytes),
            )
            # New session so kill() hits child's process group
            os.setsid()
        except Exception as e:  # noqa: BLE001
            # Never raise here — the parent cannot recover cleanly after fork
            sys.stderr.write(f"[sandbox] limit setup failed: {e}\n")


def _parse_last_json_object(text: str) -> Optional[dict[str, Any]]:
    """
    Scan backwards for the last top-level {...} block that parses as JSON.
    Lets miner code print debug lines before the result.
    """
    end = len(text)
    while True:
        close = text.rfind("}", 0, end)
        if close == -1:
            return None
        depth = 0
        start = None
        for i in range(close, -1, -1):
            ch = text[i]
            if ch == "}":
                depth += 1
            elif ch == "{":
                depth -= 1
                if depth == 0:
                    start = i
                    break
        if start is None:
            return None
        candidate = text[start : close + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        end = start  # scan earlier in the text
