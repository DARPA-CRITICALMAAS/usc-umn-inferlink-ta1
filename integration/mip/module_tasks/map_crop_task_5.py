# Copyright 2024 InferLink Corporation

from mip.utils.docker_task import DockerTask
from mip.module_tasks.legend_segment_task_2 import LegendSegmentTask2
from mip.utils.checker import check_directory_exists
from mip.module_tasks.registry import register_task


@register_task
class MapCropTask5(DockerTask):
    NAME = "map_crop"
    REQUIRES = [LegendSegmentTask2]

    def run_post(self):
        d = self.task_config.host_task_output_dir

        # TODO: match this to the path/stride params
        check_directory_exists(path=d / f"{self.map_name}_g256_s256", min_files=1)
        check_directory_exists(path=d / f"{self.map_name}_g1000_s1000", min_files=1)
