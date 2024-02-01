# Copyright 2024 InferLink Corporation

from mip.utils.docker_task import DockerTask
from mip.module_tasks.text_spotting_task_6 import TextSpottingTask6
from mip.utils.checker import check_file_exists
from mip.module_tasks.registry import register_task


@register_task
class LegendItemSegmentTask3(DockerTask):
    NAME = "legend_item_segment"

    def requires(self):
        return [
            TextSpottingTask6(job_name=self.config.job_name, map_name=self.config.map_name),
        ]

    def run_post(self):
        d = self.task_config.host_task_output_dir
        check_file_exists(d / f"{self.map_name}_PointLineType.geojson")
