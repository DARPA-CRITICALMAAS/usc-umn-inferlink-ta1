# Copyright 2024 InferLink Corporation

from pathlib import Path

from docker_task import DockerTask
from legend_item_segment_task_3 import LegendItemSegmentTask3
from registry import register


class LegendItemDescriptionTask4(DockerTask):
    NAME = "legend_item_description"

    def requires(self):
        return [
            LegendItemSegmentTask3(job_id=self.job_id, job_name=self.job_name, _config=self.config),
        ]

    def _get_expected_outputs(self) -> tuple[list[Path], list[Path]]:
        return (
            [
                Path(self.job_name + "_point.json"),
                Path(self.job_name + "_line.json"),
                Path(self.job_name + "_polygon.json"),
            ],
            []
        )

register(LegendItemDescriptionTask4)
