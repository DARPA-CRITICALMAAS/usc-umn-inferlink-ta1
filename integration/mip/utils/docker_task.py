# Copyright 2024 InferLink Corporation

import datetime
import logging
from typing import TextIO

from mip.utils.docker_runner import DockerRunner
from mip.utils.simple_task import SimpleTask

logger = logging.getLogger('luigi-interface')


class DockerTask(SimpleTask):

    NAME = "invalid"
    GPU = True
    USER = "cmaas"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run_body(self):

        docker_log_path = self.task_config.host_output_dir / f"{self.NAME}.docker.txt"

        container = self._make_container()

        logger.debug("-----------------------------------------------")
        logger.debug(container.shell_command)
        logger.debug("")
        logger.debug(container.run_command)
        logger.debug("-----------------------------------------------")

        with open(docker_log_path, "w") as f:
            print(container.shell_command, file=f)
            print(container.run_command, file=f)

        status, log_data, elapsed = container.run(self.perf_collector)


        with open(docker_log_path, "a") as f:
            print(log_data, file=f)
            print("\n", file=f)
            self._print_perf(f)
            print(f"# elapsed: {elapsed} seconds", file=f)
            print(f"# exit_status: {status}", file=f)

        logger.debug("-----------------------------------------------")
        logger.debug(log_data)
        logger.debug("-----------------------------------------------")

        if status:
            raise Exception(f"docker run failed: {self.NAME}")

    def _print_perf(self, f: TextIO) -> None:
        gb = 1024 * 1024 * 1024

        host_data, container_data = self.perf_collector.get_peak_data()

        host_mem_used = round(host_data.mem_used / gb, 1)
        host_mem_avail = round(self.perf_collector.host_total_mem / gb, 1)
        host_num_cpus = self.perf_collector.host_num_cpus
        host_num_gpus = self.perf_collector.host_num_gpus
        host_cpu_util = round(host_data.cpu_util)
        host_gpu_util = round(host_data.gpu_util)
        host_total_cpu = host_num_cpus * 100
        host_total_gpu = host_num_gpus * 100

        container_mem_used = round(container_data.mem_used / gb, 1)
        container_mem_avail = round(self.perf_collector.container_total_mem / gb, 1)
        container_num_cpus = self.perf_collector.container_num_cpus
        container_num_gpus = self.perf_collector.container_num_gpus
        container_cpu_util = round(container_data.cpu_util)
        container_gpu_util = round(container_data.gpu_util)
        container_total_cpu = container_num_cpus * 100
        container_total_gpu = container_num_gpus * 100

        print("\n", file=f)
        print(f"# host peak_mem_used: {host_mem_used} GB", file=f)
        print(f"# host total_mem: {host_mem_avail} GB", file=f)
        print(f"# host peak_cpu: {host_cpu_util}%", file=f)
        print(f"# host total_cpu: {host_total_cpu}%", file=f)
        print(f"# host peak_gpu: {host_gpu_util}%", file=f)
        print(f"# host total_gpu: {host_total_gpu}%", file=f)

        print("\n", file=f)
        print(f"# container peak_mem_used: {container_mem_used} GB", file=f)
        print(f"# container total_mem: {container_mem_avail} GB", file=f)
        print(f"# container peak_cpu: {container_cpu_util}%", file=f)
        print(f"# container total_cpu: {container_total_cpu}%", file=f)
        print(f"# container peak_gpu: {container_gpu_util}%", file=f)
        print(f"# container total_gpu: {container_total_gpu}%", file=f)

    def _make_container(self) -> DockerRunner:
        image_name = f"inferlink/ta1_{self.NAME}"
        gpus = self.GPU

        environment = [
            f"OPENAI_API_KEY={self.config.openai_key}"
        ]

        user = self.USER

        volumes = [
            f"{self.task_config.host_input_dir}:{self.task_config.container_input_dir}",
            f"{self.task_config.host_output_dir}:{self.task_config.container_output_dir}",
            f"{self.task_config.host_temp_dir}:{self.task_config.container_temp_dir}",
        ]

        options = self.task_config.get_options()

        container_name = f"{self.job_name}__{self.NAME}_{datetime.datetime.now().strftime('%H%M%S')}"

        container = DockerRunner(
            image=image_name,
            name=container_name,
            command=options,
            volumes=volumes,
            environment=environment,
            user=user,
            gpus=gpus,
        )

        return container
