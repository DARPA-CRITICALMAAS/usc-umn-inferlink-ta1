# Copyright 2024 InferLink Corporation

import luigi

from registry import register
from simple_task import SimpleTask
from start_task_0 import StartTask0
from map_segment_task_1 import MapSegmentTask1
from legend_segment_task_2 import LegendSegmentTask2
from legend_item_segment_task_3 import LegendItemSegmentTask3
from legend_item_description_task_4 import LegendItemDescriptionTask4
from map_crop_task_5 import MapCropTask5
from text_spotting_task_6 import TextSpottingTask6
from line_extract_task_7 import LineExtractTask7
from polygon_extract_task_8 import PolygonExtractTask8


class EndTask(SimpleTask):
    NAME = "end"

    def requires(self):
        return [
            StartTask0(job_id=self.job_id, job_name=self.job_name, _config=self.config),
            MapSegmentTask1(job_id=self.job_id, job_name=self.job_name, _config=self.config),
            LegendSegmentTask2(job_id=self.job_id, job_name=self.job_name, _config=self.config),
            LegendItemSegmentTask3(job_id=self.job_id, job_name=self.job_name, _config=self.config),
            LegendItemDescriptionTask4(job_id=self.job_id, job_name=self.job_name, _config=self.config),
            MapCropTask5(job_id=self.job_id, job_name=self.job_name, _config=self.config),
            TextSpottingTask6(job_id=self.job_id, job_name=self.job_name, _config=self.config),
            LineExtractTask7(job_id=self.job_id, job_name=self.job_name, _config=self.config),
            PolygonExtractTask8(job_id=self.job_id, job_name=self.job_name, _config=self.config),
        ]

    def output(self):
        return luigi.LocalTarget(f"{self.host_job_dir}/{self.NAME}.txt")

    def run(self):
        with self.output().open('w') as f:
            f.write(f"{self.job_id} {self.job_name}")


register(EndTask)
