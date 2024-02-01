# Copyright 2024 InferLink Corporation

import argparse
import os
from pathlib import Path
from typing import Any, Optional

import yaml

DEFAULT_CONFIG_FILE = "config.yml"


class Options:
    def __init__(self):

        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_FILE)
        parser.add_argument("--map", type=str)
        parser.add_argument("--job", type=str)
        parser.add_argument("--target-task")
        parser.add_argument("--list-tasks", action="store_true")

        args = parser.parse_args()

        self.map_name: str = args.map
        self.job_name: str = args.job
        self.target_task_name: str = args.target_task
        self.config_file: str = args.config
        self.list_tasks: bool = args.list_tasks

        if not self.list_tasks:
            if not self.map_name:
                parser.error("--map-name is required")
            if not self.job_name:
                parser.error("--job-name is required")
