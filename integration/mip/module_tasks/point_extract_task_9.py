# Copyright 2024 InferLink Corporation

from mip.utils.docker_task import DockerTask
from mip.module_tasks.legend_item_description_task_4 import LegendItemDescriptionTask4
from mip.module_tasks.map_crop_task_5 import MapCropTask5
from mip.utils.checker import check_directory_exists
from mip.module_tasks.registry import register_task


@register_task
class PointExtractTask9(DockerTask):
    NAME = "point_extract"

    def requires(self):
        return [
            LegendItemDescriptionTask4(job_name=self.config.job_name, map_name=self.config.map_name),
            MapCropTask5(job_name=self.config.job_name, map_name=self.config.map_name),
        ]

    def run_post(self):
        # TODO
        d = self.task_config.host_task_output_dir