# Copyright 2024 InferLink Corporation

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml

from tasker.utils.options import Options


class Config:

    # TODO: hack, because we don't know how to pass non-string state to Tasks
    CONFIG: Optional[Config] = None

    def __init__(self, options: Options):

        self.map_name: str = options.map_name
        self.job_name: str = options.job_name
        self.target_task_name: str = options.target_task_name

        config_text = Path(options.config_file).read_text()
        self.data = yaml.load(config_text, Loader=yaml.FullLoader)

        self.openai_key = Path(f"{os.path.expanduser('~')}/.ssh/openai").read_text().strip()

        self.host_input_dir = Path(self.data["host"]["input_dir"])
        self.host_output_dir = Path(self.data["host"]["output_dir"])
        self.host_temp_dir = Path(self.data["host"]["temp_dir"])
        self.container_input_dir = Path(self.data["container"]["input_dir"])
        self.container_output_dir = Path(self.data["container"]["output_dir"])
        self.container_temp_dir = Path(self.data["container"]["temp_dir"])

        self.host_job_output_dir = self.host_output_dir / self.job_name
        self.host_job_temp_dir = self.host_temp_dir / self.job_name

        Config.CONFIG = self

class TaskConfig:
    def __init__(self, config: Config, task_name: str):
        self._config = config
        self.task_name = task_name

        self.container_input_dir = config.container_input_dir
        self.container_output_dir = config.container_output_dir
        self.container_temp_dir = config.container_temp_dir

        self.container_task_output_dir = self.container_output_dir / task_name
        self.container_task_temp_dir = self.container_temp_dir / task_name

        self.host_input_dir = config.host_input_dir
        self.host_output_dir = config.host_job_output_dir
        self.host_temp_dir = config.host_job_temp_dir

        self.host_task_output_dir = self.host_output_dir / task_name
        self.host_task_temp_dir = self.host_temp_dir / task_name

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
            s = s.replace("$OUTPUT_DIR", str(self.container_output_dir))
            s = s.replace("$TEMP_DIR", str(self.container_temp_dir))
            return s
        return str(s)
