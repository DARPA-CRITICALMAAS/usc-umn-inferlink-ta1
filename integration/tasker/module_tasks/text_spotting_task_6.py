# Copyright 2024 InferLink Corporation

from tasker.utils.docker_task import DockerTask
from tasker.module_tasks.map_crop_task_5 import MapCropTask5
from tasker.utils.checker import check_directory_exists


class TextSpottingTask6(DockerTask):
    NAME = "text_spotting"
    USER = "root"
    GPU = True

    def requires(self):
        return [
            MapCropTask5(job_name=self.config.job_name, map_name=self.config.map_name),
        ]

    def run_post(self):
        d = self.task_config.host_task_output_dir
        check_directory_exists(d / "mapKurator_test" / "spotter" / "test")
