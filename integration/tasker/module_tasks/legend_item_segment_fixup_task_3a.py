# Copyright 2024 InferLink Corporation

import shutil

from tasker.utils.simple_task import SimpleTask
from tasker.module_tasks.legend_item_segment_task_3 import LegendItemSegmentTask3
from tasker.module_tasks.registry import register_task


@register_task
class LegendItemSegmentFixupTask3a(SimpleTask):
    NAME = "legend_item_segment_fixup"

    def requires(self):
        return [
            LegendItemSegmentTask3(job_name=self.config.job_name, map_name=self.config.map_name),
        ]

    def run_body(self):
        src = (self.task_config.host_input_dir
               / "input"
               / self.config.map_name
               / "legend_item_segment"
               / f"{self.config.map_name}.json")
        dst = self.task_config.host_output_dir / self.task_config.task_name / "output"
        shutil.copyfile(src, dst)
