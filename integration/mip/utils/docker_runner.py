# Copyright 2024 InferLink Corporation

from pathlib import Path
import requests  # needed for docker exceptions
import time
from typing import Any, Optional

import docker
import docker.types
import docker.errors


class DockerRunner:
    def __init__(self, *,
                 name: str,
                 image: str,
                 command: list[str],
                 volumes: list[str],  # [/host:/container]
                 environment: list[str],  # [VAR=value]
                 log_file: Path,
                 gpus: bool,
                 user: Optional[str] = None):

        self.mem_gb_used = 0
        self.mem_gb_avail = 0
        self.cpu_perc = 0
        self.cpu_max_perc = 0

        self._log_file = log_file

        self._client = docker.from_env()

        mounts = [_make_mount(v) for v in volumes]

        if gpus:
            device = docker.types.DeviceRequest(
                driver="nvidia",
                count=-1,
                capabilities=[["gpu"]])
            devices = [device]
        else:
            devices = []

        try:
            c = self._client.containers.get(name)
            c.remove()
        except docker.errors.NotFound:
            pass

        # filter out "", used for switches with no values
        command = [c for c in command if c]

        self._container = self._client.containers.create(
            image=image,
            name=name,
            mounts=mounts,
            command=command,
            environment=environment,
            user=user,
            device_requests=devices,
        )

        vs = ""
        for v in volumes:
            vs += f" -v {v}"

        options = ""
        for c in command:
            options += f" {c}"

        gpus_s = "--gpus all" if gpus else ""
        self.shell_command = f"# docker run {gpus_s} --user {user} {vs} -it --entrypoint bash {image}\n"
        self.run_command = f"# docker run {gpus_s} --user {user} {vs} {image} {options}\n"

        print()
        print("-----------------------------------------------")
        print(self.shell_command)
        print()
        print(self.run_command)
        print("-----------------------------------------------")
        print()

        with open(self._log_file, "w") as f:
            print(self.shell_command, file=f)
            print(self.run_command, file=f)

    # returns (status code, log string)
    def run(self) -> tuple[int, str, int]:
        start = time.time()

        self._container.start()

        exit_status = self._wait_for_completion()

        end = time.time()
        elapsed = round(end-start)

        print(f"# elapsed: {elapsed} seconds")
        print(f"# exit_status: {exit_status}")

        log = self._container.logs(stdout=True, stderr=True)
        log = log.decode("utf-8")
        print(log)

        with open(self._log_file, "a") as f:
            print(log, file=f)
            print("\n", file=f)
            print(f"# exit_status: {exit_status}", file=f)
            print(f"# elapsed: {elapsed} seconds", file=f)
            print(f"# peak_mem_used: {self.mem_gb_used} GB", file=f)
            print(f"# max_mem: {self.mem_gb_avail} GB", file=f)
            print(f"# peak_cpu: {self.cpu_perc}%", file=f)
            print(f"# max_cpu: {self.cpu_max_perc}%", file=f)

        return exit_status, log, elapsed

    def _wait_for_completion(self) -> int:
        # use the wait(timeout) call a perf stats collector (and potential heartbeat)
        while True:
            try:
                exit_status = self._container.wait(timeout=1)
                return exit_status["StatusCode"]
            except requests.exceptions.ConnectionError as ex:
                if "read timed out" in str(ex).lower():
                    pass
            self._update_perf()
        # not reached

    def _update_perf(self) -> None:
        stats: dict[str, Any] = self._container.stats(decode=False, stream=False)
        if not stats:
            return

        # sometimes the fields are empty...
        try:
            mem_bytes_used = stats["memory_stats"]["usage"]
            mem_bytes_avail = stats["memory_stats"]["limit"]
            mem_gb_used = round(mem_bytes_used / (1024 * 1024 * 1024), 1)
            mem_gb_avail = round(mem_bytes_avail / (1024 * 1024 * 1024), 1)
            self.mem_gb_used = max(self.mem_gb_used, mem_gb_used)
            self.mem_gb_avail = max(self.mem_gb_avail, mem_gb_avail)
        except KeyError:
            pass

        try:
            cpu_usage = (stats['cpu_stats']['cpu_usage']['total_usage']
                         - stats['precpu_stats']['cpu_usage']['total_usage'])
            cpu_system = (stats['cpu_stats']['system_cpu_usage']
                          - stats['precpu_stats']['system_cpu_usage'])
            num_cpus = stats['cpu_stats']["online_cpus"]
            cpu_perc = round((cpu_usage / cpu_system) * num_cpus * 100)
            cpu_max_perc = num_cpus * 100

            self.cpu_perc = max(self.cpu_perc, cpu_perc)
            self.cpu_max_perc = max(self.cpu_max_perc, cpu_max_perc)
        except KeyError:
            pass


def _make_mount(v: str) -> docker.types.Mount:
    t = v.split(":")
    mount = docker.types.Mount(
        type='bind',
        source=t[0],
        target=t[1],
        read_only=False)
    return mount
