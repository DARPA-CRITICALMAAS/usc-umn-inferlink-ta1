# Copyright 2024 InferLink Corporation

from datetime import datetime
from typing import Any, Optional

from docker.models.containers import Container
import psutil


class PerfRecord:
    def __init__(
            self,
            *,
            timestamp: Optional[datetime],
            cpu_util: float,  # percentage, [0..n*100]
            mem_used: int,  # bytes
            gpu_util: float,  # percentage, [0..n*100]
            gpu_mem_used: int,
    ):  # bytes
        self.timestamp = timestamp
        self.cpu_util = cpu_util
        self.mem_used = mem_used
        self.gpu_util = gpu_util
        self.gpu_mem_used = gpu_mem_used

    def __str__(self) -> str:
        t = f"[{self.timestamp.strftime('%H:%M:%S')}] " if self.timestamp else ""
        mem_gb = round(self.mem_used / (1024 * 1024 * 1024), 1)
        gpu_mem_gb = round(self.gpu_mem_used / (1024 * 1024 * 1024), 1)
        return f"{t}cpu={self.cpu_util}% mem={mem_gb}GB gpu={self.gpu_util}% gpu_mem={gpu_mem_gb}GB"


class PerfStats:

    def __init__(self):
        self._data: list[PerfRecord] = list()

        self._num_cpus: int = 0
        self._total_mem: int = 0
        self._num_gpus: int = 0
        self._total_gpu_mem: int = 0

    def get_peak_data(self) -> PerfRecord:
        cpu_util = max([p.cpu_util for p in self._data])
        mem_used = max([p.mem_used for p in self._data])
        gpu_util = max([p.gpu_util for p in self._data])
        gpu_mem_used = max([p.gpu_mem_used for p in self._data])
        data = PerfRecord(
            timestamp=None,
            cpu_util=cpu_util,
            mem_used=mem_used,
            gpu_util=gpu_util,
            gpu_mem_used=gpu_mem_used)

        return data

    def update_for_host(self) -> Optional[PerfRecord]:
        mem = psutil.virtual_memory()
        mem_bytes_used = mem.total - mem.available
        mem_bytes_total = mem.total

        cpu_perc = psutil.cpu_percent()
        num_cpus = psutil.cpu_count()

        if not self._total_mem:
            self._total_mem = mem_bytes_total
        if not self._num_cpus:
            self._num_cpus = num_cpus

        data = PerfRecord(
            timestamp=datetime.now(),
            cpu_util=cpu_perc,
            mem_used=mem_bytes_used,
            gpu_util=0.0,
            gpu_mem_used=0)

        self._data.append(data)
        return data

    def update_for_container(self, container: Container) -> Optional[PerfRecord]:
        stats: dict[str, Any] = container.stats(decode=False, stream=False)
        if not stats:
            return None

        # sometimes the fields are empty...
        mem_bytes_used = 0
        mem_bytes_total = 0
        num_cpus = 0.0
        cpu_perc = 0.0

        try:
            mem_bytes_used = stats["memory_stats"]["usage"]
            mem_bytes_total = stats["memory_stats"]["limit"]
            cpu_usage = (stats['cpu_stats']['cpu_usage']['total_usage']
                         - stats['precpu_stats']['cpu_usage']['total_usage'])
            cpu_system = (stats['cpu_stats']['system_cpu_usage']
                          - stats['precpu_stats']['system_cpu_usage'])
            num_cpus = stats['cpu_stats']["online_cpus"]
            cpu_perc = round((cpu_usage / cpu_system) * num_cpus * 100.0)
        except KeyError:
            pass

        if not self._total_mem:
            self._total_mem = mem_bytes_total
        if not self._num_cpus:
            self._num_cpus = num_cpus

        data = PerfRecord(
            timestamp=datetime.now(),
            cpu_util=cpu_perc,
            mem_used=mem_bytes_used,
            gpu_util=0.0,
            gpu_mem_used=0)

        self._data.append(data)
        return data

    def dump_report(self) -> str:
        s = ""

        def sprint(m: str):
            nonlocal s
            s += m + "\n"

        gb = 1024 * 1024 * 1024

        data = self.get_peak_data()

        num_cpus = self._num_cpus
        cpu_util = round(data.cpu_util)
        total_cpu = num_cpus * 100

        mem_used = round(data.mem_used / gb, 1)
        mem_avail = round(self._total_mem / gb, 1)

        num_gpus = self._num_gpus
        gpu_util = round(data.gpu_util)
        total_gpu = num_gpus * 100

        gpu_mem_used = round(data.gpu_mem_used / gb, 1)
        gpu_mem_avail = round(self._total_gpu_mem / gb, 1)

        sprint(f"# peak_cpu: {cpu_util}%")
        sprint(f"# total_cpu: {total_cpu}%")
        sprint(f"# peak_mem_used: {mem_used} GB")
        sprint(f"# total_mem: {mem_avail} GB")
        sprint(f"# peak_gpu: {gpu_util}%")
        sprint(f"# total_gpu: {total_gpu}%")
        sprint(f"# peak_gpu_mem_used: {gpu_mem_used} GB")
        sprint(f"# total_gpu_mem: {gpu_mem_avail} GB")

        return s


class PerfCollector:

    def __init__(self):
        self._host_perf_stats = PerfStats()
        self._container_perf_stats = PerfStats()

    def update(self, container: Optional[Container] = None) -> tuple[Optional[PerfRecord], Optional[PerfRecord]]:
        host_data = self._host_perf_stats.update_for_host()

        if container:
            container_data = self._container_perf_stats.update_for_container(container)
        else:
            container_data = None

        return host_data, container_data

    def dump_report(self) -> str:
        s = ""

        s += "\n"
        s += "HOST PERF\n"
        s += self._host_perf_stats.dump_report()

        s += "\n"
        s += "CONTAINER PERF\n"
        s += self._container_perf_stats.dump_report()

        return s
