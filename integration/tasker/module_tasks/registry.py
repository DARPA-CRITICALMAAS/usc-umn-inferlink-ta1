# Copyright 2024 InferLink Corporation

from typing import Callable, Optional

from tasker.module_tasks.start_task_0 import StartTask0
from tasker.module_tasks.legend_segment_task_2 import LegendSegmentTask2
from tasker.module_tasks.legend_item_segment_task_3 import LegendItemSegmentTask3
from tasker.module_tasks.legend_item_description_task_4 import LegendItemDescriptionTask4
from tasker.module_tasks.map_crop_task_5 import MapCropTask5
from tasker.module_tasks.text_spotting_task_6 import TextSpottingTask6
from tasker.module_tasks.line_extract_task_7 import LineExtractTask7
from tasker.module_tasks.polygon_extract_task_8 import PolygonExtractTask8
from tasker.module_tasks.point_extract_task_9 import PointExtractTask9
from tasker.module_tasks.georeference_task_10 import GeoreferenceTask10
from tasker.module_tasks.end_task import EndTask


_TASKS: dict[str, Callable] = {
    StartTask0.NAME: StartTask0,
    LegendSegmentTask2.NAME: LegendSegmentTask2,
    LegendItemSegmentTask3.NAME: LegendItemSegmentTask3,
    LegendItemDescriptionTask4.NAME: LegendItemDescriptionTask4,
    MapCropTask5.NAME: MapCropTask5,
    TextSpottingTask6.NAME: TextSpottingTask6,
    LineExtractTask7.NAME: LineExtractTask7,
    PolygonExtractTask8.NAME: PolygonExtractTask8,
    PointExtractTask9.NAME: PointExtractTask9,
    GeoreferenceTask10.NAME: GeoreferenceTask10,
    EndTask.NAME: EndTask,
}


def registry_lookup(name: str) -> Optional[Callable]:
    return _TASKS.get(name, None)
