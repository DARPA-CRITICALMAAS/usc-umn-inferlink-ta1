# Copyright 2024 InferLink Corporation

import argparse
import os
from pathlib import Path
from typing import Any, Optional

import yaml

DEFAULT_CONFIG_FILE = "config.yml"


class Config:
    def __init__(self):

        parser = argparse.ArgumentParser()
        parser.add_argument("--config-file", type=str, default=DEFAULT_CONFIG_FILE)
        parser.add_argument("--map-name", type=str, required=True)
        parser.add_argument("--job-name", type=str, required=True)
        parser.add_argument("--target-task-name", type=str, required=True)

        args = parser.parse_args()

        self.map_name: str = args.map_name
        self.job_name: str = args.job_name
        self.target_task_name: str = args.target_task_name

        config_text = Path(args.config_file).read_text()
        self.data = yaml.load(config_text, Loader=yaml.FullLoader)

        self.openai_key = Path(f"{os.path.expanduser('~')}/.ssh/openai").read_text().strip()

        self.host_input_dir = Path(self.data["host"]["input_dir"])
        self.host_output_dir = Path(self.data["host"]["output_dir"])
        self.host_temp_dir = Path(self.data["host"]["temp_dir"])
        self.container_input_dir = Path(self.data["container"]["input_dir"])
        self.container_output_dir = Path(self.data["container"]["output_dir"])
        self.container_temp_dir = Path(self.data["container"]["temp_dir"])

        self.container_job_output_dir = self.container_output_dir / self.job_name
        self.container_job_temp_dir = self.container_temp_dir / self.job_name

        self.host_job_output_dir = self.host_output_dir / self.job_name
        self.host_job_temp_dir = self.host_temp_dir / self.job_name


class TaskConfig:
    def __init__(self, config: Config, task_name: str):
        self._config = config
        self.task_name = task_name

        self.container_input_dir = config.container_input_dir
        self.container_job_output_dir = config.container_job_output_dir
        self.container_job_temp_dir = config.container_job_temp_dir
        self.container_task_output_dir = config.container_job_output_dir / task_name
        self.container_task_temp_dir = config.container_job_temp_dir / task_name

        self.host_input_dir = config.host_input_dir
        self.host_job_output_dir = config.host_job_output_dir
        self.host_job_temp_dir = config.host_job_temp_dir
        self.host_task_output_dir = config.host_job_output_dir / task_name
        self.host_task_temp_dir = config.host_job_temp_dir / task_name

    def get_options(self) -> list[str]:
        ret: list[str] = []
        for k, v in self._config.data[self.task_name].items():
            ret.append(f"--{k}")
            if type(v) is list:
                for vi in v:
                    if vi:
                        ret.append(f"{self._expand(vi)}")
            else:
                if v:
                    ret.append(f"{self._expand(v)}")
        return ret

    def _expand(self, s: Any) -> str:
        if type(s) is str:
            s = s.replace("$MAP_NAME", self._config.map_name)
            s = s.replace("$INPUT_DIR", str(self.container_input_dir))
            s = s.replace("$OUTPUT_DIR", str(self.container_job_output_dir))
            s = s.replace("$TEMP_DIR", str(self.container_job_temp_dir))
            return s
        return str(s)


CONFIG: Optional[Config] = None
