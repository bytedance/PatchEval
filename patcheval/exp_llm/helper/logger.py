# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import sys
import os
import threading
from pathlib import Path
from typing import Optional

class CustomFormatter(logging.Formatter):
    """Custom formatter to enrich log records with CVE and relative path info."""

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "cve"):
            record.cve = "GLOBAL"
        if hasattr(record, "pathname"):
            record.relativepath = os.path.relpath(record.pathname)
        else:
            record.relativepath = "unknown"
        return super().format(record)


def setup_logger(
    log_file_path: str,
    log_level: int = logging.DEBUG,
    debug_mode: bool = False,
    overwrite: bool = True,
) -> logging.Logger:
    """
    Setup and return the global logger.

    Parameters:
    - log_file_path: path to the log file.
    - log_level: logging level for file and console.
    - debug_mode: whether to output logs to stdout in addition to file.
    - overwrite: always open the file in write mode to avoid growing too large.
    """
    logger = logging.getLogger("GlobalLogger")
    logger.setLevel(log_level)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    # Ensure directory exists
    log_file = f"{log_file_path}"
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    # Formatter
    log_format = CustomFormatter(
        "%(asctime)s - %(relativepath)s:%(lineno)d - %(levelname)s - [%(cve)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, mode="w" if overwrite else "a", encoding="utf-8")
    file_handler.setFormatter(log_format)
    file_handler.setLevel(log_level)
    logger.addHandler(file_handler)

    # Optional console handler
    if debug_mode:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        console_handler.setLevel(log_level)
        logger.addHandler(console_handler)

    # Avoid double propagation
    logger.propagate = False
    return logger


class RelativePathFilter(logging.Filter):
    """Filter to convert absolute path to relative path against current working directory."""

    def filter(self, record: logging.LogRecord) -> bool:
        script_dir = os.getcwd()
        abs_path = record.pathname
        record.relativepath = os.path.relpath(abs_path, script_dir)
        return True


def get_logger() -> logging.Logger:
    """Return the configured global logger."""
    logger = logging.getLogger("GlobalLogger")
    logger.addFilter(RelativePathFilter())
    logger.propagate = False
    return logger


class ContextualLogger:
    """Per-task contextual logger that appends LogRecord to a bound buffer.

    This logger preserves user-visible log strings and formatting by creating
    logging.LogRecord objects identical to those produced by the previous
    `_cache_log` implementation, while removing repetitive parameters at call
    sites. The buffer reference is shared across the task, and records are
    flushed by the caller using `base_logger.handle(record)`.

    Args:
        base_logger: The configured global logger obtained via `get_logger()`.
        cve_id: The CVE identifier to bind as record.extra['cve'].
        buffer_ref: A list to which LogRecord objects are appended.
        allowed_levels: Optional iterable of logging levels to allow. If provided,
            only records with levels in this set will be appended.
    """

    def __init__(self, base_logger: logging.Logger, cve_id: str, buffer_ref: list, allowed_levels: Optional[set] = None):
        self._base_logger = base_logger
        self._cve_id = cve_id
        self._buffer_ref = buffer_ref
        self._allowed_levels = set(allowed_levels or [])

    def _emit(self, level: int, msg: str, extra: Optional[dict] = None) -> None:
        import inspect
        # Only honor allowed levels filtering to preserve prior shim behavior
        if self._allowed_levels and level not in self._allowed_levels:
            return
        extra = dict(extra or {})
        extra["cve"] = self._cve_id
        stack = inspect.stack()
        if len(stack) >= 3:
            frame = stack[2]
            filename = frame.filename
            lineno = frame.lineno
        else:
            filename = __file__
            lineno = 0
        record = self._base_logger.makeRecord(
            name=self._base_logger.name,
            level=level,
            fn=filename,
            lno=lineno,
            msg=msg,
            args=(),
            exc_info=None,
            func=None,
            extra=extra,
        )
        self._buffer_ref.append(record)

    def debug(self, msg: str, extra: Optional[dict] = None) -> None:
        self._emit(logging.DEBUG, msg, extra)

    def info(self, msg: str, extra: Optional[dict] = None) -> None:
        self._emit(logging.INFO, msg, extra)

    def warning(self, msg: str, extra: Optional[dict] = None) -> None:
        self._emit(logging.WARNING, msg, extra)

    def error(self, msg: str, extra: Optional[dict] = None) -> None:
        self._emit(logging.ERROR, msg, extra)


class LogContextManager:
    """Thread-safe manager for per-task contextual loggers.

    Provides start/bind/get/finalize operations keyed by `task_id` (commonly a CVE id).
    The manager is concurrency-safe and designed for multi-thread scenarios.
    """

    def __init__(self, base_logger: Optional[logging.Logger] = None):
        self._base_logger = base_logger or get_logger()
        self._lock = threading.Lock()  # mutex for contexts
        self._contexts = {}
        # contexts: task_id -> {"logger": ContextualLogger, "buffer": list, "levels": set, "cve": str}
        self._tls = threading.local()  # thread-local binding for current task

    def start_task(self, task_id: str, cve_id: str, buffer_ref: list, allowed_levels: Optional[set] = None) -> None:
        with self._lock:
            self._contexts[task_id] = {
                "logger": ContextualLogger(self._base_logger, cve_id, buffer_ref, allowed_levels),
                "buffer": buffer_ref,
                "levels": set(allowed_levels or []),
                "cve": cve_id,
            }
        # Bind current task in thread-local for immediate use
        try:
            self._tls.current_task_id = task_id
        except Exception:
            pass

    def bind_current_task(self, task_id: str) -> None:
        """Bind the current thread to the given task_id for contextual logging."""
        self._tls.current_task_id = task_id

    def unbind_current_task(self) -> None:
        """Unbind current task from thread-local without removing contexts."""
        if hasattr(self._tls, "current_task_id"):
            try:
                del self._tls.current_task_id
            except Exception:
                # Silently ignore to preserve behavior
                self._tls.current_task_id = None

    def get_logger(self, task_id: str) -> ContextualLogger:
        ctx = self._contexts.get(task_id)
        if not ctx:
            # Fallback: bind to a no-op buffer to avoid crashes; caller should start_task first
            dummy_buffer = []
            ctx = {
                "logger": ContextualLogger(self._base_logger, task_id, dummy_buffer, None),
                "buffer": dummy_buffer,
                "levels": set(),
                "cve": task_id,
            }
            with self._lock:
                self._contexts[task_id] = ctx
        return ctx["logger"]

    def get_current_logger(self) -> ContextualLogger:
        """Return the ContextualLogger bound to the current thread via `bind_current_task`/`start_task`.

        If no task is currently bound, returns a logger bound to a dummy buffer
        so that logging calls are no-ops but do not crash.
        """
        task_id = getattr(self._tls, "current_task_id", None)
        if task_id is None:
            # Return a dummy contextual logger
            dummy_buffer = []
            return ContextualLogger(self._base_logger, "GLOBAL", dummy_buffer, None)
        return self.get_logger(task_id)

    def finalize_task(self, task_id: str) -> list:
        """Clear logger context mapping for the task and return the bound buffer.

        Note: Does NOT flush records automatically to keep behavior identical
        to prior implementation where flush happens explicitly at CVE end.
        """
        with self._lock:
            ctx = self._contexts.pop(task_id, None)
        # Ensure thread-local binding is cleared if pointing to this task
        if getattr(self._tls, "current_task_id", None) == task_id:
            self.unbind_current_task()
        return ctx.get("buffer", []) if ctx else []

    def clear_contexts(self) -> None:
        with self._lock:
            self._contexts.clear()
            # Also clear thread-local binding
            self.unbind_current_task()


