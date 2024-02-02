# Copyright 2024 InferLink Corporation

from __future__ import annotations

from datetime import datetime
import logging
import shutil
from typing import Optional

import luigi

from mip.utils.config import Config
from mip.utils.task_config import TaskConfig
from mip.utils.perf_collector import PerfCollector


logger = logging.getLogger('luigi-interface')


class SimpleTask(luigi.Task):
    NAME = "__simple__"
    REQUIRES: list[type[SimpleTask]] = None

    job_name = luigi.Parameter()
    map_name = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config = Config.CONFIG
        self.task_config = TaskConfig(self.config, self.NAME)
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

        self.perf_collector = PerfCollector()

    def run(self):
        self.start_time = datetime.now()

        for p in [self.task_config.host_task_output_dir, self.task_config.host_task_temp_dir]:
            shutil.rmtree(p, ignore_errors=True)
            p.mkdir(parents=False, exist_ok=False)

        try:
            self.run_pre()
        except Exception as ex:
            logger.error(f"run_pre() failed: {self.task_config.task_name}", exc_info=ex)
            raise

        logger.info(f"run_pre() completed: {self.task_config.task_name}")

        try:
            self.run_body()
        except Exception as ex:
            logger.error(f"run_body() failed: {self.task_config.task_name}", exc_info=ex)
            raise

        logger.info(f"run_body() completed: {self.task_config.task_name}")

        try:
            self.run_post()
        except Exception as ex:
            logger.error(f"run_post() failed: {self.task_config.task_name}", exc_info=ex)
            raise

        logger.info(f"run_post() completed: {self.task_config.task_name}")

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

    def requires(self):
        # force each class to list their predecessors
        assert self.REQUIRES is not None

        return [
            cls(job_name=self.config.job_name, map_name=self.config.map_name)
            for cls in self.REQUIRES
        ]
