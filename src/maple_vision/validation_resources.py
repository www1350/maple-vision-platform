"""Bounded process measurements for local wall-clock validation.

CPU includes all threads in this process, not its children. Current resident
memory and lifetime peak resident memory deliberately remain separate fields.
Unsupported or inaccessible counters are unknown, never silently zero.
"""

import math
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProcessMemory:
    current_rss_bytes: int | None
    lifetime_peak_rss_bytes: int | None
    current_rss_source: str
    lifetime_peak_rss_source: str
    unavailable_reason: str | None = None


def read_process_memory() -> ProcessMemory:
    """Read actual process counters; do not label tracemalloc as total RSS."""
    current: int | None = None
    peak: int | None = None
    current_source = "unavailable"
    peak_source = "unavailable"
    reasons: list[str] = []
    if sys.platform.startswith("linux"):
        try:
            fields = Path("/proc/self/statm").read_text(encoding="ascii").split()
            pages = int(fields[1])
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            if pages < 0 or page_size <= 0:
                raise ValueError("invalid resident page counter")
            current = pages * page_size
            current_source = "linux:/proc/self/statm:resident_pages*SC_PAGE_SIZE"
        except (OSError, ValueError, IndexError):
            reasons.append("current RSS counter unavailable")
    else:
        reasons.append("current RSS counter is not implemented on this platform")
    if sys.platform.startswith("linux") or sys.platform == "darwin":
        try:
            import resource

            native_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if not math.isfinite(native_peak) or native_peak < 0:
                raise ValueError("invalid lifetime peak counter")
            multiplier = 1 if sys.platform == "darwin" else 1024
            peak = int(native_peak * multiplier)
            native_unit = "bytes" if multiplier == 1 else "KiB"
            peak_source = f"resource.getrusage(RUSAGE_SELF).ru_maxrss:{native_unit}"
        except (ImportError, OSError, ValueError):
            reasons.append("lifetime peak RSS counter unavailable")
    else:
        reasons.append("lifetime peak RSS counter is not implemented on this platform")
    return ProcessMemory(current, peak, current_source, peak_source, "; ".join(reasons) or None)


class ProcessResourceMonitor:
    """Sample independently of episode length and retain a bounded time series.

    On overflow, deterministic decimation retains the first and latest sample and
    approximately halves older points. Aggregate extrema include discarded samples.
    The sampling interval is a requested cadence, not a real-time scheduling promise.
    """

    def __init__(self, interval_seconds: float = 1.0, max_samples: int = 1024) -> None:
        if (
            isinstance(interval_seconds, bool)
            or not isinstance(interval_seconds, (int, float))
            or not math.isfinite(interval_seconds)
            or not 0.05 <= interval_seconds <= 60
        ):
            raise ValueError("resource interval must be finite in [0.05, 60] seconds")
        if type(max_samples) is not int or not 2 <= max_samples <= 10_000:
            raise ValueError("max resource samples must be an integer in [2, 10000]")
        self.interval_seconds = float(interval_seconds)
        self.max_samples = max_samples
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[dict[str, Any]] = []
        self._measurements = 0
        self._discarded = 0
        self._max_rss: int | None = None
        self._max_cpu_percent: float | None = None
        self._wall_start = 0.0
        self._cpu_start = 0.0
        self._previous_wall = 0.0
        self._previous_cpu = 0.0
        self._error: str | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise ValueError("resource monitor cannot be restarted")
        self._wall_start = self._previous_wall = time.monotonic()
        self._cpu_start = self._previous_cpu = time.process_time()
        self.sample()
        self._thread = threading.Thread(
            target=self._poll, name="maple-resource-sampler", daemon=True
        )
        self._thread.start()

    def _poll(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self.sample()
            except Exception as exc:
                # A missing counter cannot turn an otherwise real soak into a
                # fabricated complete resource record. Preserve the explicit error.
                with self._lock:
                    self._error = f"{type(exc).__name__}: resource sample failed"
                return

    def sample(self) -> None:
        """Record one measurement; also used for the start and final endpoints."""
        with self._lock:
            wall, cpu = time.monotonic(), time.process_time()
            elapsed = wall - self._wall_start
            cpu_elapsed = cpu - self._cpu_start
            if elapsed < 0 or cpu_elapsed < 0:
                raise ValueError("resource measurement clocks moved backwards")
            wall_delta = wall - self._previous_wall
            cpu_delta = cpu - self._previous_cpu
            percent = 100 * cpu_delta / wall_delta if wall_delta > 0 else None
            memory = read_process_memory()
            row = {
                "elapsed_wall_seconds": round(elapsed, 9),
                "process_cpu_seconds": round(cpu_elapsed, 9),
                "cpu_percent_since_previous": None if percent is None else round(percent, 6),
                **asdict(memory),
            }
            self._previous_wall, self._previous_cpu = wall, cpu
            self._measurements += 1
            self._samples.append(row)
            if memory.current_rss_bytes is not None:
                self._max_rss = max(self._max_rss or 0, memory.current_rss_bytes)
            if percent is not None:
                self._max_cpu_percent = max(self._max_cpu_percent or 0, percent)
            if len(self._samples) > self.max_samples:
                retained = [self._samples[0], *self._samples[2::2]]
                if retained[-1] is not self._samples[-1]:
                    retained.append(self._samples[-1])
                self._discarded += len(self._samples) - len(retained)
                self._samples = retained

    def close(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                raise RuntimeError("resource monitor did not stop")
        self.sample()
        return self.report()

    def report(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema_version": 1,
                "scope": "current_process_all_threads_excluding_children",
                "wall_clock_source": "time.monotonic",
                "cpu_source": "time.process_time",
                "units": {
                    "elapsed_wall_seconds": "seconds",
                    "process_cpu_seconds": "seconds_since_monitor_start",
                    "cpu_percent_since_previous": "one_logical_cpu_percent; can_exceed_100",
                    "current_rss_bytes": "bytes",
                    "lifetime_peak_rss_bytes": "bytes; process_lifetime_not_soak_only",
                },
                "requested_interval_seconds": self.interval_seconds,
                "max_samples": self.max_samples,
                "measurements": self._measurements,
                "retained_samples": len(self._samples),
                "discarded_samples": self._discarded,
                "retention": "bounded_decimation_preserves_first_and_latest",
                "max_observed_current_rss_bytes": self._max_rss,
                "max_observed_cpu_percent": self._max_cpu_percent,
                "sampling_error": self._error,
                "samples": [dict(row) for row in self._samples],
                "limitations": [
                    "Sampling may miss short peaks; cadence depends on OS scheduling.",
                    "Current RSS and process lifetime peak RSS are different counters.",
                    "Unsupported memory counters remain null with their reason.",
                    "CPU and memory include other work in this process, not only inference.",
                ],
            }
