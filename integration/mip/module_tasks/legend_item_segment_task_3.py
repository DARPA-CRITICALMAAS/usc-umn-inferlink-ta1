# Copyright 2024 InferLink Corporation

import shutil

from mip.utils.docker_task import DockerTask
from mip.module_tasks.text_spotting_task_6 import TextSpottingTask6
from mip.utils.checker import check_file_exists
from mip.module_tasks.registry import register_task


@register_task
class LegendItemSegmentTask3(DockerTask):
    NAME = "legend_item_segment"
    REQUIRES = [TextSpottingTask6]

    def run_post(self):
        d = self.task_config.host_task_output_dir
        check_file_exists(d / f"{self.map_name}_PointLineType.geojson")

        # TODO: remove this fixup step
        src = (self.task_config.host_input_dir
               / "input"
               / self.config.map_name
               / "legend_item_segment"
               / f"{self.config.map_name}.json")
        dst = self.task_config.host_output_dir / self.task_config.task_name / "output"
        shutil.copyfile(src, dst)
