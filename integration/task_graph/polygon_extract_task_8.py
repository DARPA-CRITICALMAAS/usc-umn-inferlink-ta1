# Copyright 2024 InferLink Corporation

from pathlib import Path

from docker_task import DockerTask
from line_extract_task_7 import LineExtractTask7
from registry import register


class PolygonExtractTask8(DockerTask):
    NAME = "polygon_extract"

    def requires(self):
        return [
            LineExtractTask7(job_id=self.job_id, job_name=self.job_name, _config=self.config),
        ]

    def _get_expected_outputs(self) -> tuple[list[Path], list[Path]]:
        return (
            [],
            []
        )


register(PolygonExtractTask8)
