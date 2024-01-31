# Copyright 2024 InferLink Corporation

from typing import cast

import luigi


class MyDict:
    def __init__(self, d):
        self.data = d


class SimpleTask(luigi.Task):
    job_id = luigi.Parameter()
    job_name = luigi.Parameter()
    _config = luigi.Parameter(significant=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.container_data_dir = self.config.data["config"]["DATA_DIR"]
        self.container_job_dir = self.config.data["config"]["JOB_DIR"]
        self.host_data_dir = self.config.data["config"]["HOST_DATA_DIR"]
        self.host_job_dir = self.config.data["config"]["HOST_JOB_DIR"]

    def run(self):
        pass

    @property
    def config(self) -> MyDict:
        return cast(MyDict, self._config)
