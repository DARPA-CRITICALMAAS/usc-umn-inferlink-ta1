# Copyright 2024 InferLink Corporation

from tasker.utils.docker_task import DockerTask
from tasker.module_tasks.legend_segment_task_2 import LegendSegmentTask2
from tasker.utils.checker import check_directory_exists
from tasker.module_tasks.registry import register_task


@register_task
class MapCropTask5(DockerTask):
    NAME = "map_crop"

    def requires(self):
        return [
            LegendSegmentTask2(job_name=self.config.job_name, map_name=self.config.map_name),
        ]

    def run_post(self):
        d = self.task_config.host_task_output_dir

        # TODO: match this to the path/stride params
        check_directory_exists(path=d / f"{self.map_name}_g256_s256", min_files=1)
        check_directory_exists(path=d / f"{self.map_name}_g1000_s1000", min_files=1)
