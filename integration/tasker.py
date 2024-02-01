#!/usr/bin/env python3
# Copyright 2024 InferLink Corporation

import sys

import luigi

from tasker.module_tasks.registry import registry_lookup
import tasker.utils.config as config


# datetime.datetime.now().strftime("%H.%M.%S")


def main() -> int:
    cfg = config.Config()
    config.CONFIG = cfg

    for p in [cfg.host_job_output_dir, cfg.host_job_temp_dir]:
        p.mkdir(parents=True, exist_ok=True)

    task_cls = registry_lookup(cfg.target_task_name)
    if not task_cls:
        print(f"task not found: {cfg.target_task_name}")
        return 1

    result = luigi.build(
        tasks=[task_cls(job_name=cfg.job_name, map_name=cfg.map_name)],
        local_scheduler=True,
        detailed_summary=True
    )

    return 0 if result.status == luigi.execution_summary.LuigiStatusCode.SUCCESS else 1


if __name__ == '__main__':
    status = main()
    sys.exit(status)