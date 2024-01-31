# Copyright 2024 InferLink Corporation

from pathlib import Path

from docker_task import DockerTask
from map_crop_task_5 import MapCropTask5
from registry import register


class TextSpottingTask6(DockerTask):
    NAME = "text_spotting"

    def requires(self):
        return [
            MapCropTask5(job_id=self.job_id, job_name=self.job_name, _config=self.config),
        ]

    def _get_expected_outputs(self) -> tuple[list[Path], list[Path]]:
        return (
            [],
            [Path("mapKurator_test/spotter/test")]
        )

register(TextSpottingTask6)
