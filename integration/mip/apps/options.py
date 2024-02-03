# Copyright 2024 InferLink Corporation

import argparse
import os
from pathlib import Path


DEFAULT_CONFIG_FILE = "./config.yml"
DEFAULT_TASK_NAME = "all"
DEFAULT_OPENAI_KEY_FILE = f"{os.path.expanduser('~')}/.ssh/openai"


class Options:
    def __init__(self):

        parser = argparse.ArgumentParser(
            prog="mipper",
            description="Runs TA1 modules in an integrated fashion")

        parser.add_argument(
            "--config-file", "-c",
            type=str,
            default=DEFAULT_CONFIG_FILE,
            help=f"path to YML configuration file (default: {DEFAULT_CONFIG_FILE})",
        )
        parser.add_argument(
            "--map-name", "-i",
            type=str,
            help="name of map (example: WY_CO_Peach)",
        )
        parser.add_argument(
            "--job-name", "-j",
            type=str,
            help="name of job to execute",
        )
        parser.add_argument(
            "--module-name", "-m",
            type=str,
            action='extend',
            nargs='*',
            help="name of target module to run (may be repeated)",
        )
        parser.add_argument(
            "--list-modules",
            action="store_true",
            help="list names of known modules and exit"
        )
        parser.add_argument(
            "--list-deps",
            action="store_true",
            help="display module dependency tree and exit"
        )
        parser.add_argument(
            "--openai_key_file",
            type=str,
            default=DEFAULT_OPENAI_KEY_FILE,
            help=f"path to file containing OpenAI key (default: {DEFAULT_OPENAI_KEY_FILE})"
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="forces execution of target module, even if already completed successfully"
        )

        args = parser.parse_args()
        self.map_name: str = args.map_name
        self.job_name: str = args.job_name
        self.target_task_names: list[str] = args.module_name
        self.config_file: str = args.config_file
        self.list_tasks: bool = args.list_modules
        self.list_deps: bool = args.list_deps
        self.openai_key_file = Path(args.openai_key_file)
        self.force = args.force

        if not self.target_task_names:
            self.target_task_names = [DEFAULT_TASK_NAME]

        if not self.list_tasks:
            if not self.map_name:
                parser.error("--map-name is required")
            if not self.job_name:
                parser.error("--job-name is required")
