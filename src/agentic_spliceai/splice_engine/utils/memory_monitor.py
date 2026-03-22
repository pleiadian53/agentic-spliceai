"""Memory monitoring utility for long-running workflows.

Provides a lightweight background thread that periodically checks process
RSS (Resident Set Size) and logs warnings when memory usage approaches a
configurable limit.  Can optionally raise an exception to allow the caller
to checkpoint and exit gracefully before the OS OOM-killer strikes.

Usage as a context manager
--------------------------
>>> with MemoryMonitor(limit_gb=4.0, interval_seconds=30) as mon:
...     for chrom in chromosomes:
...         process(chrom)
...         mon.check()  # optional inline check (also runs in background)

Usage with FeatureWorkflow
--------------------------
>>> monitor = MemoryMonitor(limit_gb=4.0)
>>> monitor.start()
>>> try:
...     result = workflow.run(chromosomes=chromosomes)
... finally:
...     monitor.stop()
"""

import logging
import os
import platform
import resource
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class MemoryLimitExceeded(Exception):
    """Raised when process RSS exceeds the configured critical threshold."""

    def __init__(self, rss_gb: float, limit_gb: float) -> None:
        self.rss_gb = rss_gb
        self.limit_gb = limit_gb
        super().__init__(
            f"Memory limit exceeded: RSS={rss_gb:.2f} GB > limit={limit_gb:.2f} GB"
        )


@dataclass
class MemorySnapshot:
    """A point-in-time memory measurement."""

    rss_bytes: int
    timestamp: float

    @property
    def rss_mb(self) -> float:
        return self.rss_bytes / (1024 * 1024)

    @property
    def rss_gb(self) -> float:
        return self.rss_bytes / (1024 * 1024 * 1024)


def get_rss_bytes() -> int:
    """Get current process RSS in bytes (cross-platform)."""
    if platform.system() == "Darwin":
        # macOS: ru_maxrss is in bytes
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    else:
        # Linux: ru_maxrss is in kilobytes
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def get_rss_bytes_precise() -> int:
    """Get current RSS from /proc/self/status (Linux) or ps (macOS).

    More accurate than ru_maxrss which reports peak, not current RSS.
    Falls back to ru_maxrss if platform-specific method fails.
    """
    try:
        if platform.system() == "Linux":
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) * 1024  # KB → bytes
        elif platform.system() == "Darwin":
            # Use ps to get current RSS (not peak)
            pid = os.getpid()
            import subprocess

            out = subprocess.check_output(
                ["ps", "-p", str(pid), "-o", "rss="],
                text=True,
                timeout=5,
            )
            return int(out.strip()) * 1024  # KB → bytes
    except Exception:
        pass

    return get_rss_bytes()


def get_system_memory_bytes() -> int:
    """Get total system physical memory in bytes."""
    if platform.system() == "Darwin":
        import subprocess

        out = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], text=True, timeout=5
        )
        return int(out.strip())
    elif platform.system() == "Linux":
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024  # KB → bytes
    # Fallback
    return 16 * 1024 * 1024 * 1024  # assume 16 GB


class MemoryMonitor:
    """Background memory monitor with configurable thresholds.

    Parameters
    ----------
    limit_gb : float, optional
        Hard memory limit in GB.  When RSS exceeds this, the monitor
        sets the ``exceeded`` flag and optionally raises on the next
        ``check()`` call.  Default: 80% of system RAM.
    warn_fraction : float
        Fraction of ``limit_gb`` at which to start logging warnings.
        Default 0.75 (warn at 75% of limit).
    interval_seconds : float
        How often the background thread checks RSS.  Default 30s.
    abort_on_exceed : bool
        If True, ``check()`` raises ``MemoryLimitExceeded`` when the
        limit is exceeded.  The background thread only logs — it never
        raises.  Default True.
    """

    def __init__(
        self,
        limit_gb: float | None = None,
        warn_fraction: float = 0.75,
        interval_seconds: float = 30.0,
        abort_on_exceed: bool = True,
    ) -> None:
        if limit_gb is None:
            system_gb = get_system_memory_bytes() / (1024**3)
            limit_gb = round(system_gb * 0.8, 1)
            logger.info(
                "MemoryMonitor: system RAM=%.1f GB, limit set to %.1f GB (80%%)",
                system_gb,
                limit_gb,
            )

        self.limit_gb = limit_gb
        self.limit_bytes = int(limit_gb * 1024 * 1024 * 1024)
        self.warn_bytes = int(self.limit_bytes * warn_fraction)
        self.interval_seconds = interval_seconds
        self.abort_on_exceed = abort_on_exceed

        self._exceeded = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_rss_bytes = 0
        self._lock = threading.Lock()

    @property
    def exceeded(self) -> bool:
        """Whether the memory limit has been exceeded."""
        return self._exceeded.is_set()

    @property
    def peak_rss_gb(self) -> float:
        """Peak RSS observed during monitoring, in GB."""
        with self._lock:
            return self._peak_rss_bytes / (1024**3)

    def snapshot(self) -> MemorySnapshot:
        """Take a memory snapshot and update peak tracking."""
        rss = get_rss_bytes_precise()
        with self._lock:
            self._peak_rss_bytes = max(self._peak_rss_bytes, rss)
        return MemorySnapshot(rss_bytes=rss, timestamp=time.time())

    def check(self) -> MemorySnapshot:
        """Check current memory and raise if limit exceeded.

        Call this at safe checkpointing boundaries (e.g., between
        chromosomes) so the workflow can save progress before exiting.

        Returns
        -------
        MemorySnapshot
            Current memory state.

        Raises
        ------
        MemoryLimitExceeded
            If ``abort_on_exceed`` is True and RSS > limit.
        """
        snap = self.snapshot()

        if snap.rss_bytes > self.limit_bytes:
            self._exceeded.set()
            if self.abort_on_exceed:
                raise MemoryLimitExceeded(snap.rss_gb, self.limit_gb)

        return snap

    def start(self) -> "MemoryMonitor":
        """Start the background monitoring thread."""
        if self._thread is not None and self._thread.is_alive():
            return self

        self._stop_event.clear()
        self._exceeded.clear()
        self._thread = threading.Thread(
            target=self._monitor_loop,
            name="MemoryMonitor",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "MemoryMonitor started: limit=%.1f GB, warn=%.1f GB, "
            "interval=%ds, abort=%s",
            self.limit_gb,
            self.warn_bytes / (1024**3),
            self.interval_seconds,
            self.abort_on_exceed,
        )
        return self

    def stop(self) -> None:
        """Stop the background monitoring thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info(
            "MemoryMonitor stopped. Peak RSS: %.2f GB", self.peak_rss_gb
        )

    def _monitor_loop(self) -> None:
        """Background loop — logs periodically, never raises."""
        while not self._stop_event.wait(self.interval_seconds):
            snap = self.snapshot()

            if snap.rss_bytes > self.limit_bytes:
                self._exceeded.set()
                logger.critical(
                    "Memory CRITICAL: RSS=%.2f GB > limit=%.2f GB — "
                    "workflow should checkpoint and exit",
                    snap.rss_gb,
                    self.limit_gb,
                )
            elif snap.rss_bytes > self.warn_bytes:
                logger.warning(
                    "Memory WARNING: RSS=%.2f GB (%.0f%% of %.1f GB limit)",
                    snap.rss_gb,
                    100 * snap.rss_bytes / self.limit_bytes,
                    self.limit_gb,
                )
            else:
                logger.info(
                    "Memory OK: RSS=%.2f GB (%.0f%% of %.1f GB limit)",
                    snap.rss_gb,
                    100 * snap.rss_bytes / self.limit_bytes,
                    self.limit_gb,
                )

    # Context manager protocol
    def __enter__(self) -> "MemoryMonitor":
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
