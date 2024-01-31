# Copyright 2024 InferLink Corporation

from pathlib import Path

from docker_task import DockerTask
from start_task_0 import StartTask0
from registry import register


class MapSegmentTask1(DockerTask):
    NAME = "map_segment"

    def _get_expected_outputs(self) -> tuple[list[Path], list[Path]]:
        return (
            [self.job_name + ".geojson", self.job_name + ".tif"],
            []
        )

    def requires(self):
        return [
            StartTask0(job_id=self.job_id, job_name=self.job_name, _config=self.config),
        ]


register(MapSegmentTask1)
