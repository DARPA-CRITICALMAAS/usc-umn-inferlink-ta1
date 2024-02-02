# Copyright 2024 InferLink Corporation

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from docker.models.containers import Container
import psutil


@dataclass
class PerfData:
    timestamp: Optional[datetime]
    cpu_util: float  # percentage, [0..n*100]
    mem_used: int  # bytes
    gpu_util: float  # percentage, [0..n*100]

    def __str__(self) -> str:
        t = f"[{self.timestamp.strftime('%H:%M:%S')}] " if self.timestamp else ""
        mem_gb = round(self.mem_used / (1024 * 1024 * 1024), 1)
        return f"{t}cpu={self.cpu_util}% mem={mem_gb}GB gpu={self.gpu_util}%"


class PerfCollector:

    def __init__(self):
        self.host_data: list[PerfData] = list()
        self.container_data: list[PerfData] = list()

        self.host_total_mem: int = 0
        self.host_num_cpus: int = 0
        self.host_num_gpus: int = 0
        self.container_total_mem: int = 0
        self.container_num_cpus: int = 0
        self.container_num_gpus: int = 0

    def get_peak_data(self) -> tuple[PerfData, PerfData]:
        cpu_util = max([p.cpu_util for p in self.host_data])
        mem_used = max([p.mem_used for p in self.host_data])
        gpu_util = max([p.gpu_util for p in self.host_data])
        host_data = PerfData(None, cpu_util, mem_used, gpu_util)

        cpu_util = max([p.cpu_util for p in self.container_data])
        mem_used = max([p.mem_used for p in self.container_data])
        gpu_util = max([p.gpu_util for p in self.container_data])
        container_data = PerfData(None, cpu_util, mem_used, gpu_util)

        return host_data, container_data

    def update(self, container: Optional[Container] = None) -> tuple[PerfData, PerfData]:
        host_data = self._get_host_perf()
        self.host_data.append(host_data)

        if container:
            container_data = self._get_container_perf(container)
            self.container_data.append(container_data)
        else:
            container_data = PerfData(datetime.now(), 0.0, 0, 0.0)

        return host_data, container_data

    def _get_host_perf(self) -> PerfData:
        mem = psutil.virtual_memory()
        mem_bytes_used = mem.total - mem.available
        mem_bytes_total = mem.total

        cpu_perc = psutil.cpu_percent()
        num_cpus = psutil.cpu_count()

        if not self.host_total_mem:
            self.host_total_mem = mem_bytes_total
        if not self.host_num_cpus:
            self.host_num_cpus = num_cpus

        data = PerfData(datetime.now(), cpu_perc, mem_bytes_used, 0.0)
        return data

    def _get_container_perf(self, container: Container) -> Optional[PerfData]:
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

        if not self.container_total_mem:
            self.container_total_mem = mem_bytes_total
        if not self.container_num_cpus:
            self.container_num_cpus = num_cpus

        data = PerfData(datetime.now(), cpu_perc, mem_bytes_used, 0.0)
        return data
