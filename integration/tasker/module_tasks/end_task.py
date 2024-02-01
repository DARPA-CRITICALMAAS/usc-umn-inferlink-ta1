# Copyright 2024 InferLink Corporation

from tasker.utils.simple_task import SimpleTask
from tasker.module_tasks.start_task_0 import StartTask0
from tasker.module_tasks.legend_segment_task_2 import LegendSegmentTask2
from tasker.module_tasks.legend_item_segment_task_3 import LegendItemSegmentTask3
from tasker.module_tasks.legend_item_description_task_4 import LegendItemDescriptionTask4
from tasker.module_tasks.map_crop_task_5 import MapCropTask5
from tasker.module_tasks.text_spotting_task_6 import TextSpottingTask6
from tasker.module_tasks.line_extract_task_7 import LineExtractTask7
from tasker.module_tasks.polygon_extract_task_8 import PolygonExtractTask8


class EndTask(SimpleTask):
    NAME = "end"

    def requires(self):
        return [
            StartTask0(job_name=self.config.job_name, map_name=self.config.map_name, config=self.config),
            LegendSegmentTask2(job_name=self.config.job_name, map_name=self.config.map_name, config=self.config),
            LegendItemSegmentTask3(job_name=self.config.job_name, map_name=self.config.map_name, config=self.config),
            LegendItemDescriptionTask4(job_name=self.config.job_name, map_name=self.config.map_name, config=self.config),
            MapCropTask5(job_name=self.config.job_name, map_name=self.config.map_name, config=self.config),
            TextSpottingTask6(job_name=self.config.job_name, map_name=self.config.map_name, config=self.config),
            # LineExtractTask7(job_name=self.config.job_name, map_name=self.config.map_name, config=self.config),
            # PolygonExtractTask8(job_name=self.config.job_name, map_name=self.config.map_name, config=self.config),
        ]
