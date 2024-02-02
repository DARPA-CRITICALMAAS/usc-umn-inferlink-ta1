# Copyright 2024 InferLink Corporation

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml

from mip.apps.options import Options


class Config:

    # TODO: hack, because we don't know how to pass non-string state to Tasks
    CONFIG: Optional[Config] = None

    def __init__(self, options: Options):

        self.map_name: str = options.map_name
        self.job_name: str = options.job_name

        config_text = Path(options.config_file).read_text()
        self.data = yaml.load(config_text, Loader=yaml.FullLoader)

        self.openai_key = options.openai_key_file.read_text().strip()

        self.host_input_dir = Path(self.data["host"]["input_dir"])
        self.host_output_dir = Path(self.data["host"]["output_dir"])
        self.host_temp_dir = Path(self.data["host"]["temp_dir"])
        self.container_input_dir = Path(self.data["container"]["input_dir"])
        self.container_output_dir = Path(self.data["container"]["output_dir"])
        self.container_temp_dir = Path(self.data["container"]["temp_dir"])

        self.host_job_output_dir = self.host_output_dir / self.job_name
        self.host_job_temp_dir = self.host_temp_dir / self.job_name

        Config.CONFIG = self
