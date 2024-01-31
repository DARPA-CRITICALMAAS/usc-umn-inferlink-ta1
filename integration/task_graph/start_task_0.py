# Copyright 2024 InferLink Corporation

import luigi

from simple_task import SimpleTask


class StartTask0(SimpleTask):
    NAME = "start"

    def output(self):
        return luigi.LocalTarget(f"{self.host_job_dir}/{self.NAME}_task.txt")

    def run(self):
        with self.output().open('w') as f:
            f.write(f"{self.job_id} {self.job_name}")
