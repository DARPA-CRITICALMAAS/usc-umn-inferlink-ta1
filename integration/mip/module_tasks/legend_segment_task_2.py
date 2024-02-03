# Copyright 2024 InferLink Corporation

from mip.utils.docker_task import DockerTask
from mip.module_tasks.start_task_0 import StartTask0
from mip.utils.checker import check_file_exists
from mip.module_tasks.registry import register_task


@register_task
class LegendSegmentTask2(DockerTask):
    NAME = "legend_segment"
    REQUIRES = [StartTask0]

    def run_post(self):
        d = self.task_config.host_task_output_dir / f"{self.map_name}_map_segmentation.json"
        check_file_exists(d)
