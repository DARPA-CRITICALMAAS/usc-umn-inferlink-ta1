# Copyright 2024 InferLink Corporation

from mip.utils.docker_task import DockerTask
from mip.module_tasks.registry import register_task
from mip.module_tasks.start_task_0 import StartTask0
from mip.utils.checker import check_directory_exists


@register_task
class GeoreferenceTask10(DockerTask):
    NAME = "georeference"

    def requires(self):
        return [
            StartTask0(job_name=self.config.job_name, map_name=self.config.map_name),
        ]

    def run_post(self):
        # TODO
        d = self.task_config.host_task_output_dir