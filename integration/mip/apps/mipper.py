#!/usr/bin/env python3
# Copyright 2024 InferLink Corporation

import sys

import luigi
import luigi.tools.deps_tree as deps_tree
import nvidia_smi

from mip.module_tasks import *  # force registration of all module tasks
from mip.module_tasks.registry import registry_lookup, get_tasks
from mip.utils.config import Config
from mip.utils.task_config import TaskConfig
from mip.utils.simple_task import SimpleTask
from mip.apps.options import Options


def main() -> int:
    opts = Options()

    if opts.list_tasks:
        print("Registered tasks:")
        for name, cls in get_tasks().items():
            if cls.REQUIRES:
                s = ", ".join([c.NAME for c in cls.REQUIRES])
            else:
                s = "[]"
            print(f"    {name}  <--  {s}")
        return 0

    if not opts.openai_key_file.exists():
        print(f"OpenAI key file not found: {opts.openai_key_file}")
        return 1

    cfg = Config(opts)

    for p in [cfg.host_input_dir, cfg.host_output_dir, cfg.host_temp_dir]:
        if not p.exists():
            print(f"host directory not found: {p}")
            return 1
        if not p.is_dir():
            print(f"host directory is not a directory: {p}")
            return 1

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

    if opts.force:
        for task_name in opts.target_task_names:
            task_config = TaskConfig(cfg, task_name)
            task_config.host_task_file.unlink(missing_ok=True)

    nvidia_smi.nvmlInit()

    result = luigi.build(
        tasks=tasks,
        local_scheduler=True,
        detailed_summary=True
    )

    status = 0 if result.status == luigi.execution_summary.LuigiStatusCode.SUCCESS else 1

    nvidia_smi.nvmlShutdown()

    return status


if __name__ == '__main__':
    status = main()
    sys.exit(status)
