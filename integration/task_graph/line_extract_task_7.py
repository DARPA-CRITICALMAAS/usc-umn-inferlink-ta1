# Copyright 2024 InferLink Corporation

from pathlib import Path

from docker_task import DockerTask
from legend_item_description_task_4 import LegendItemDescriptionTask4
from registry import register


class LineExtractTask7(DockerTask):
    NAME = "line_extract"

    def requires(self):
        return [
            LegendItemDescriptionTask4(job_id=self.job_id, job_name=self.job_name, _config=self.config),
        ]

    def _get_expected_outputs(self) -> tuple[list[Path], list[Path]]:
        return (
            [],
            []
        )


register(LineExtractTask7)
