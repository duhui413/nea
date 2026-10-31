"""One OS lock prevents duplicate paid calls for a phase."""
from __future__ import annotations
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_GUARDS: dict[str, threading.RLock] = {}
_GUARDS_LOCK = threading.Lock()


class RunBusyError(RuntimeError):
    """Raised when another process already owns a run-directory lock."""

class RunDirectoryLock:
    """A non-blocking OS advisory lock for one experiment output directory.

    Runs under ``outputs`` rendezvous in the sibling ``.runtime/locks`` tree,
    preserving the run's relative path. Other locations retain their adjacent
    ``.locks`` directory. Existing legacy lock files are also acquired so an
    older process still excludes a new one during the transition. Lock files
    may remain after release; ownership is the OS-level lock on the first byte,
    which the operating system releases if a process exits or crashes.
    """

    def __init__(self, run_directory: Path):
        self.run_directory = run_directory.resolve()
        self._legacy_path = self.run_directory.parent / ".locks" / f"{self.run_directory.name}.lock"
        output_root = next((path for path in self.run_directory.parents
                            if path.name.casefold() == "outputs"), None)
        self.path = (output_root.parent / ".runtime" / "locks"
                     / self.run_directory.parent.relative_to(output_root)
                     / f"{self.run_directory.name}.lock"
                     if output_root is not None else self._legacy_path)
        self._handles: list[Any] = []
        self._held = False

    def acquire(self) -> None:
        """Take the lock immediately, or raise a clear non-retryable busy error."""

        if self._held:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handles = []
        try:
            handles.append(self._acquire_file(self.path, create=True))
            if self._legacy_path != self.path:
                # Opening without creation is intentional: a new run must not
                # recreate an artifact-side .locks directory. If cleanup has
                # already removed an old file, only the new rendezvous applies.
                try:
                    legacy = self._acquire_file(self._legacy_path, create=False)
                except FileNotFoundError:
                    pass
                else:
                    handles.append(legacy)
        except BaseException:
            for handle in reversed(handles):
                handle.close()  # Closing also releases a successfully held lock.
            raise
        self._handles = handles
        self._held = True

    def _acquire_file(self, path: Path, *, create: bool) -> Any:
        handle = path.open("a+b" if create else "r+b")
        try:
            # ``msvcrt.locking`` cannot lock past EOF, whereas an empty POSIX
            # lock file is fine.  A stable byte also makes both implementations
            # use exactly the same region.
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
        except BaseException:
            handle.close()
            raise
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise RunBusyError(
                f"run directory is busy: {self.run_directory} (lock: {path})"
            ) from exc
        return handle

    def release(self) -> None:
        """Release an owned lock; process exit also releases an unreleased one."""

        if not self._held:
            return
        handles, self._handles = self._handles, []
        self._held = False
        error = None
        for handle in reversed(handles):
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError as exc:
                error = error or exc
            finally:
                handle.close()
        if error is not None:
            raise error

    def __enter__(self) -> "RunDirectoryLock":
        self.acquire()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()


@contextmanager
def run_directory_guard(run_directory):
    """Serialize threads of this process on one run directory, then take the OS lock (other processes still raise)."""
    key = str(Path(run_directory).resolve())
    with _GUARDS_LOCK:
        guard = _GUARDS.setdefault(key, threading.RLock())
    with guard:
        with RunDirectoryLock(Path(run_directory)):
            yield
