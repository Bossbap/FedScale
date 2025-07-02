import logging
import os
from typing import Optional

# Third‑party log handler that supports safe concurrent writes
# (pip install concurrent-log-handler)
try:
    from concurrent_log_handler import ConcurrentRotatingFileHandler as _SafeFileHandler  # type: ignore
except ImportError:  # graceful fallback – single‑process handler
    import logging.handlers as _handlers
    _SafeFileHandler = _handlers.RotatingFileHandler  # type: ignore

import fedscale.cloud.config_parser as parser

__all__ = [
    "initiate_client_setting",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_DEF_MAX_BYTES: int = 20 * 1024 * 1024  # 20 MB per file
_DEF_BACKUP_CNT: int = 5  # keep <n> historical files

_log_dir: Optional[str] = None


def _ensure_log_dir() -> str:
    """Return (and create) the executor log directory for this run."""
    global _log_dir
    if _log_dir is not None:
        return _log_dir

    _log_dir = os.path.join(
        parser.args.log_path,
        "logs",
        parser.args.job_name,
        parser.args.time_stamp,
        "executor",
    )
    os.makedirs(_log_dir, exist_ok=True)
    return _log_dir


def _build_root_logger() -> None:
    """Configure the *root* logger exactly once.

    All modules that call ``logging.getLogger(__name__)`` will inherit these
    settings, so we do *not* call this function in libraries – only from
    FedScale’s runtime entry‑points (executor.py).
    """
    log_dir = _ensure_log_dir()
    log_path = os.path.join(log_dir, "log")

    fmt = "%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"
    date_fmt = "(%m-%d) %H:%M:%S"

    # stdout/err – useful while developing or when journalctl/syslog collects it
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(fmt, date_fmt))

    # disk – safe for many processes writing at once
    file_handler = _SafeFileHandler(
        log_path,
        mode="a",
        maxBytes=_DEF_MAX_BYTES,
        backupCount=_DEF_BACKUP_CNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter(fmt, date_fmt))

    # Root logger setup (do it once!)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(file_handler)


# ---------------------------------------------------------------------------
# Public API (FedScale imports this and calls ``initiate_client_setting``)
# ---------------------------------------------------------------------------

def initiate_client_setting() -> None:
    """Initialize logging for an **executor** process.

    Safe to call multiple times but only the first call has an effect.
    """
    if not getattr(initiate_client_setting, "_configured", False):
        _build_root_logger()
        initiate_client_setting._configured = True  # type: ignore