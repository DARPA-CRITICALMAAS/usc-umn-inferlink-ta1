# Copyright 2024 InferLink Corporation

from pathlib import Path

from docker_task import DockerTask
from map_segment_task_1 import MapSegmentTask1
from registry import register


class LegendSegmentTask2(DockerTask):
    NAME = "legend_segment"

    def requires(self):
        return [
            MapSegmentTask1(job_id=self.job_id, job_name=self.job_name, _config=self.config),
        ]

    def _get_expected_outputs(self) -> tuple[list[Path], list[Path]]:
        return (
            [Path(self.job_name + "_map_segmentation.json")],
            []
        )


register(LegendSegmentTask2)
