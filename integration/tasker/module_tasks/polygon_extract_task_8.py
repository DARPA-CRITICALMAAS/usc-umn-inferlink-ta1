# Copyright 2024 InferLink Corporation

from tasker.utils.docker_task import DockerTask
from tasker.module_tasks.line_extract_task_7 import LineExtractTask7
from tasker.module_tasks.legend_item_description_task_4 import LegendItemDescriptionTask4


class PolygonExtractTask8(DockerTask):
    NAME = "polygon_extract"

    def requires(self):
        return [
            # LineExtractTask7(ob_name=self.config.job_name, map_name=self.config.map_name, config=self.config),
            LegendItemDescriptionTask4(ob_name=self.config.job_name, map_name=self.config.map_name, config=self.config),
        ]

    def run_post(self):
        # TODO
        pass
