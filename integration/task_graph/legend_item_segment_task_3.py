# Copyright 2024 InferLink Corporation

from pathlib import Path

from docker_task import DockerTask
from text_spotting_task_6 import TextSpottingTask6
from registry import register


class LegendItemSegmentTask3(DockerTask):
    NAME = "legend_item_segment"

    def requires(self):
        return [
            TextSpottingTask6(job_id=self.job_id, job_name=self.job_name, _config=self.config),
        ]

    def _get_expected_outputs(self) -> tuple[list[Path], list[Path]]:
        return (
            [Path(self.job_name + "_PointLineType.geojson")],
            []
        )

register(LegendItemSegmentTask3)
