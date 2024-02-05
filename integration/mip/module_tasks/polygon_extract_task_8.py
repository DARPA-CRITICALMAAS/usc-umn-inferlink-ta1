# Copyright 2024 InferLink Corporation

from mip.utils.docker_task import DockerTask
from mip.module_tasks.legend_item_segment_task_3 import LegendItemSegmentTask3
from mip.module_tasks.registry import register_task


@register_task
class PolygonExtractTask8(DockerTask):
    NAME = "polygon_extract"
    REQUIRES = [
            LegendItemSegmentTask3,
        ]

    def run_post(self):
        # TODO
        pass
