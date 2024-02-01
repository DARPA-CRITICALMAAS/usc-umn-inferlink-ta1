# Copyright 2024 InferLink Corporation

from tasker.utils.docker_task import DockerTask
from tasker.module_tasks.legend_item_description_task_4 import LegendItemDescriptionTask4
from tasker.module_tasks.map_crop_task_5 import MapCropTask5
from tasker.utils.checker import check_directory_exists


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
