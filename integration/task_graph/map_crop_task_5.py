# Copyright 2024 InferLink Corporation

from pathlib import Path

from docker_task import DockerTask
from legend_segment_task_2 import LegendSegmentTask2
from registry import register


class MapCropTask5(DockerTask):
    NAME = "map_crop"

    def requires(self):
        return [
            LegendSegmentTask2(job_id=self.job_id, job_name=self.job_name, _config=self.config),
        ]

    def _get_expected_outputs(self) -> tuple[list[Path], list[Path]]:
        return (
            [],
            [Path(self.job_name + "_g1000_s1000"), Path(self.job_name + "_g256_s256")],
        )


register(MapCropTask5)
