#!/usr/bin/env python3
# Copyright 2024 InferLink Corporation

import sys

import luigi
import luigi.tools.deps_tree as deps_tree

from mip.module_tasks.registry import registry_lookup, get_task_names
from mip.module_tasks import *
from mip.utils.config import Config
from mip.utils.options import Options
from mip.utils.simple_task import SimpleTask


def main() -> int:
    opts = Options()

    if opts.list_tasks:
        print("Registered tasks:")
        for task in get_task_names():
            print(f"    {task}")
        return 0

    cfg = Config(opts)

    for p in [cfg.host_job_output_dir, cfg.host_job_temp_dir]:
        p.mkdir(parents=True, exist_ok=True)

    tasks: list[SimpleTask] = list()

    for task_name in opts.target_task_names:
        task_cls = registry_lookup(task_name)
        if not task_cls:
            print(f"task not found: {task_name}")
            return 1
        task = task_cls(job_name=cfg.job_name, map_name=cfg.map_name)
        tasks.append(task)

    if opts.list_deps:
        print()
        for task in tasks:
            s = f"TASK: {task.NAME} "
            print(s + "=" * (78-len(s)))
            print(deps_tree.print_tree(task))
            print("=" * 78)
            print()
        return 0

    result = luigi.build(
        tasks=tasks,
        local_scheduler=True,
        detailed_summary=True
    )

    return 0 if result.status == luigi.execution_summary.LuigiStatusCode.SUCCESS else 1


if __name__ == '__main__':
    status = main()
    sys.exit(status)
