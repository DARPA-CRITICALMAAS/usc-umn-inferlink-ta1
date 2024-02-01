#!/usr/bin/env python3
# Copyright 2024 InferLink Corporation

import sys

import luigi

from tasker.module_tasks.registry import registry_lookup
from tasker.utils.config import Config


# datetime.datetime.now().strftime("%H.%M.%S")


def main(config: Config) -> int:

    for p in [config.host_job_output_dir, config.host_job_temp_dir]:
        p.mkdir(parents=True, exist_ok=True)

    task_cls = registry_lookup(config.target_task_name)
    if not task_cls:
        print(f"task not found: {config.target_task_name}")
        return 1

    result = luigi.build(
        tasks=[task_cls(job_name=config.job_name, map_name=config.map_name, config=config)],
        local_scheduler=True,
        detailed_summary=True
    )

    return 0 if result.status == luigi.execution_summary.LuigiStatusCode.SUCCESS else 1


if __name__ == '__main__':
    c = Config()
    status = main(c)
    sys.exit(status)
