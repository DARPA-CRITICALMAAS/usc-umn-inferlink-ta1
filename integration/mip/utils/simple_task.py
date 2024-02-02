# Copyright 2024 InferLink Corporation

from datetime import datetime
import logging
import shutil
from typing import Optional

import luigi

from mip.utils.config import Config, TaskConfig

logger = logging.getLogger('luigi-interface')


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
        elapsed = round((self.end_time-self.start_time).total_seconds())

        logger.info("-----------------------------------------------")
        logger.info(f"task: {self.task_config.task_name}")
        logger.info(f"elapsed: {elapsed} secs")
        logger.info("-----------------------------------------------")

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
        return luigi.LocalTarget(self.task_config.host_task_file)
