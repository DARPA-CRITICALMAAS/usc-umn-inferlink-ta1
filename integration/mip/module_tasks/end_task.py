# Copyright 2024 InferLink Corporation

from mip.module_tasks.registry import register_task

from mip.utils.simple_task import SimpleTask
from mip.module_tasks.start_task_0 import StartTask0
from mip.module_tasks.legend_segment_task_2 import LegendSegmentTask2
from mip.module_tasks.legend_item_segment_task_3 import LegendItemSegmentTask3
from mip.module_tasks.legend_item_description_task_4 import LegendItemDescriptionTask4
from mip.module_tasks.map_crop_task_5 import MapCropTask5
from mip.module_tasks.text_spotting_task_6 import TextSpottingTask6
from mip.module_tasks.line_extract_task_7 import LineExtractTask7
from mip.module_tasks.polygon_extract_task_8 import PolygonExtractTask8
from mip.module_tasks.point_extract_task_9 import PointExtractTask9
from mip.module_tasks.georeference_task_10 import GeoreferenceTask10


@register_task
class EndTask(SimpleTask):
    NAME = "end"
    REQUIRES = [
        StartTask0,
        LegendSegmentTask2,
        LegendItemSegmentTask3,
        LegendItemDescriptionTask4,
        MapCropTask5,
        TextSpottingTask6,
        LineExtractTask7,
        PolygonExtractTask8,
        PointExtractTask9,
        GeoreferenceTask10,
    ]

    def run_body(self):
        pass
