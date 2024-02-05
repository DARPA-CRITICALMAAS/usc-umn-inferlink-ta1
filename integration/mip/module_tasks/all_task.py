# Copyright 2024 InferLink Corporation

from mip.module_tasks.registry import register_task

from mip.utils.simple_task import SimpleTask
from mip.module_tasks.line_extract_task_7 import LineExtractTask7
from mip.module_tasks.polygon_extract_task_8 import PolygonExtractTask8
from mip.module_tasks.point_extract_task_9 import PointExtractTask9
from mip.module_tasks.georeference_task_10 import GeoreferenceTask10


@register_task
class AllTask(SimpleTask):
    NAME = "all"
    REQUIRES = [
        LineExtractTask7,
        PolygonExtractTask8,
        PointExtractTask9,
        GeoreferenceTask10,
    ]

    def run_body(self):
        pass
