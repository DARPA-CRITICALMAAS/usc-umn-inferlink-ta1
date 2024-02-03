# Copyright 2024 InferLink Corporation

from mip.utils.docker_task import DockerTask
from mip.module_tasks.map_crop_task_5 import MapCropTask5
from mip.utils.checker import check_directory_exists
from mip.module_tasks.registry import register_task


@register_task
class TextSpottingTask6(DockerTask):
    NAME = "text_spotting"
    REQUIRES = [MapCropTask5]

    USER = "root"
    GPU = True

    def run_post(self):
        d = self.task_config.host_task_output_dir
        check_directory_exists(d / "mapKurator_test" / "spotter" / "test")
