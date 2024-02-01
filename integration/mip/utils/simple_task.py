# Copyright 2024 InferLink Corporation

from datetime import datetime
import shutil
from typing import Optional

import luigi

from mip.utils.config import Config, TaskConfig


class SimpleTask(luigi.Task):
    NAME = "__simple__"

    job_name = luigi.Parameter()
    map_name = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config = Config.CONFIG
        self.task_config = TaskConfig(self.config, self.NAME)
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

    def run(self):
        self.start_time = datetime.now()

        for p in [self.task_config.host_task_output_dir, self.task_config.host_task_temp_dir]:
            shutil.rmtree(p, ignore_errors=True)
            p.mkdir(parents=False, exist_ok=False)

        self.run_pre()

        self.run_body()

        self.run_post()

        self.end_time = datetime.now()

        with self.output().open('w') as f:
            f.write(f"job_name: {self.job_name}\n")
            f.write(f"map_name: {self.map_name}\n")
            f.write(f"start_time: {self.start_time}\n")
            f.write(f"end_time: {self.end_time}\n")
            elapsed = round((self.end_time - self.start_time).total_seconds())
            f.write(f"elapsed: {elapsed} seconds\n")

    def run_pre(self):
        pass

    # override this
    def run_body(self):
        raise NotImplementedError()

    def run_post(self):
        pass

    def output(self):
        file = self.task_config.host_output_dir / f"{self.NAME}.task.txt"
        return luigi.LocalTarget(file)
