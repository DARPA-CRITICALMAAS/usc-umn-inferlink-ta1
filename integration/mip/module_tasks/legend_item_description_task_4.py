# Copyright 2024 InferLink Corporation

from mip.module_tasks.registry import register_task
from mip.utils.docker_task import DockerTask
from mip.module_tasks.legend_item_segment_task_3 import LegendItemSegmentTask3
from mip.utils.checker import check_file_exists


@register_task
class LegendItemDescriptionTask4(DockerTask):
    NAME = "legend_item_description"
    REQUIRES = [LegendItemSegmentTask3]

    def run_pre(self):
        d = self.task_config.host_task_temp_dir

        # TODO: still needed?
        (d / "gpt4_input_dir").mkdir(parents=True, exist_ok=True)
        (d / "gpt4_temp_dir").mkdir(parents=True, exist_ok=True)

    def run_post(self):
        d = self.task_config.host_task_output_dir
        check_file_exists(d / f"{self.map_name}_point.json")
        check_file_exists(d / f"{self.map_name}_line.json")
        check_file_exists(d / f"{self.map_name}_polygon.json")
