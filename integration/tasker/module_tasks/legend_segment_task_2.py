# Copyright 2024 InferLink Corporation

from tasker.utils.docker_task import DockerTask
from tasker.module_tasks.start_task_0 import StartTask0
from tasker.utils.checker import check_file_exists
from tasker.module_tasks.registry import register_task


@register_task
class LegendSegmentTask2(DockerTask):
    NAME = "legend_segment"

    def requires(self):
        return [
            StartTask0(job_name=self.config.job_name, map_name=self.config.map_name),
        ]

    def run_post(self):
        d = self.task_config.host_task_output_dir / f"{self.map_name}_map_segmentation.json"
        check_file_exists(d)
