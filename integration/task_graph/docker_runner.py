# Copyright 2024 InferLink Corporation

from pathlib import Path
import time
from typing import Optional

import docker
import docker.types
import docker.errors


class DockerRunner:
    def __init__(self, *,
                 name: str,
                 image: str,
                 command: list[str],
                 volumes: list[str],  # [/host:/container]
                 environment: list[str], # [VAR=value]
                 log_file: Path,
                 user: Optional[str] = None):

        self._log_file = log_file

        self._client = docker.from_env()

        mounts = [_make_mount(v) for v in volumes]

        gpus = docker.types.DeviceRequest(
            driver="nvidia",
            count=-1,
            capabilities=[["gpu"]])

        try:
            c = self._client.containers.get(name)
            c.remove()
        except docker.errors.NotFound:
            pass

        self._container = self._client.containers.create(
            image=image,
            name=name,
            mounts=mounts,
            command=command,
            environment=environment,
            user=user,
            device_requests=[gpus],
        )

        vs = ""
        for v in volumes:
            vs += f" -v {v}"
        cs = ""
        for c in command:
            cs += f" {c}"

        run1 = f"# docker run --gpus all -it --entrypoint bash {vs} {image}\n"
        run2 = f"# docker run --gpus all --user {user} {vs} {image} {cs}\n"

        print("-----------------------------------------------\n")
        print(run1)
        print(run2)
        print("-----------------------------------------------\n")

        with open(self._log_file, "a") as f:
            print(run1, file=f)
            print(run2, file=f)

    # returns (status code, log string)
    def run(self) -> tuple[int, str]:
        start = time.time()
        self._container.start()

        exit_status = self._container.wait()
        end = time.time()
        print(f"elapsed: {round(end-start)} seconds")
        print(exit_status)

        log = self._container.logs(stdout=True, stderr=True)
        log = log.decode("utf-8")
        print(log)

        with open(self._log_file, "a") as f:
            print(log, file=f)
            print("\n# " + str(exit_status), file=f)
            print(f"# elapsed: {round(end-start)} seconds")

        self._container.remove()

        return exit_status['StatusCode'], log

def _make_mount(v: str) -> docker.types.Mount:
    t = v.split(":")
    mount = docker.types.Mount(
        type='bind',
        source=t[0],
        target=t[1],
        read_only=False)
    return mount
