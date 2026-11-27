"""Minimal sandbox for executing model-written code: temp dir, process group kill, rlimits.

This is defence against accidents (infinite loops, memory blow-ups, fork bombs writing huge
files), not a security boundary. It runs inside the throwaway Kaggle VM.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    import resource
except ImportError:  # pragma: no cover - windows
    resource = None

MAX_OUTPUT_CHARS = 4000
MAX_READ_CHARS = 12000

AGENT_TOOLS = [
    {"type": "function", "function": {
        "name": "bash",
        "description": "Run a bash command in the repository root and return exit code, stdout and stderr. "
                       "Use it to list files, run tests (e.g. `python -m pytest -q`) or inspect code.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "The bash command to run."}},
            "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a text file from the repository.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Path relative to the repository root."}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Create or overwrite a file with the given content.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Path relative to the repository root."},
            "content": {"type": "string", "description": "Full new file content."}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "str_replace",
        "description": "Replace an exact, unique snippet of a file with new text.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Path relative to the repository root."},
            "old_str": {"type": "string", "description": "Exact text to replace; must occur exactly once."},
            "new_str": {"type": "string", "description": "Replacement text."}},
            "required": ["path", "old_str", "new_str"]}}},
]


def _limits(cpu_seconds: int, mem_bytes: int):
    def apply():
        os.setsid()
        if resource is None:
            return
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 2**20, 64 * 2**20))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        except (ValueError, OSError):
            pass

    return apply


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n... [{len(text) - limit} chars truncated] ...\n" + text[-half:]


def run_command(cmd: list[str] | str, cwd: str, timeout: float = 20, mem_gb: float = 4.0) -> tuple[int, str]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": cwd,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "LANG": "C.UTF-8",
        "TMPDIR": cwd,
    }
    if isinstance(cmd, str):
        cmd = ["bash", "-c", cmd]
    try:
        proc = subprocess.Popen(
            cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            preexec_fn=_limits(int(timeout) + 2, int(mem_gb * 2**30)) if os.name == "posix" else None,
        )
    except OSError as e:
        return 127, f"failed to start: {e}"
    try:
        out, _ = proc.communicate(timeout=timeout)
        code = proc.returncode
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        out, _ = proc.communicate()
        code = -9
        out = (out or b"") + f"\n[timeout after {timeout}s]".encode()
    return code, _truncate((out or b"").decode("utf-8", errors="replace"))


class Workspace:
    def __init__(self, files: dict[str, str] | None = None, root: str | None = None):
        self.dir = tempfile.mkdtemp(prefix="spark_ws_", dir=root)
        for rel, content in (files or {}).items():
            self.write(rel, content)

    def _resolve(self, rel: str) -> Path:
        rel = rel.strip()
        if rel.startswith("/workspace/"):
            rel = rel[len("/workspace/"):]
        p = (Path(self.dir) / rel).resolve()
        if not str(p).startswith(str(Path(self.dir).resolve())):
            raise PermissionError(f"path escapes the repository: {rel}")
        return p

    def write(self, rel: str, content: str) -> None:
        p = self._resolve(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    def read(self, rel: str) -> str:
        return self._resolve(rel).read_text()

    def exists(self, rel: str) -> bool:
        try:
            return self._resolve(rel).exists()
        except PermissionError:
            return False

    def bash(self, command: str, timeout: float = 20) -> str:
        code, out = run_command(command, self.dir, timeout=timeout)
        return f"exit_code: {code}\n{out}"

    def call(self, name: str, args: dict) -> str:
        """Execute one tool call; every failure becomes a readable error string for the model."""
        try:
            if name == "bash":
                return self.bash(str(args["command"]))
            if name == "read_file":
                text = self.read(str(args["path"]))
                return _truncate(text, MAX_READ_CHARS)
            if name == "write_file":
                self.write(str(args["path"]), str(args.get("content", "")))
                return f"wrote {args['path']}"
            if name == "str_replace":
                path, old, new = str(args["path"]), str(args["old_str"]), str(args.get("new_str", ""))
                text = self.read(path)
                n = text.count(old)
                if n != 1:
                    return f"error: old_str occurs {n} times in {path}; it must occur exactly once"
                self.write(path, text.replace(old, new, 1))
                return f"edited {path}"
            return f"error: unknown tool {name!r}"
        except KeyError as e:
            return f"error: missing argument {e}"
        except FileNotFoundError:
            return f"error: file not found: {args.get('path')}"
        except (PermissionError, IsADirectoryError, UnicodeDecodeError, OSError) as e:
            return f"error: {e}"

    def cleanup(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


PYTEST_CMD = [sys.executable, "-m", "pytest", "-q", "-x", "--no-header", "-p", "no:cacheprovider"]


def run_tests(files: dict[str, str], test_file: str = "test_solution.py", timeout: float = 30) -> tuple[bool, str]:
    ws = Workspace(files)
    try:
        code, out = run_command(PYTEST_CMD + [test_file], ws.dir, timeout=timeout)
        return code == 0, out
    finally:
        ws.cleanup()


def run_tests_many(jobs: list[dict], workers: int | None = None, timeout: float = 30) -> list[tuple[bool, str]]:
    """jobs: [{'files': {...}, 'test_file': 'test_solution.py'}]"""
    workers = workers or max(1, min(4, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda j: run_tests(j["files"], j.get("test_file", "test_solution.py"), timeout), jobs))
