# Copyright 2024 InferLink Corporation

from abc import ABC, abstractmethod
import os
from pathlib import Path
import subprocess
from typing import Any


class Runner(ABC):
    def __init__(self, module_name: str, config: dict[str, Any]):
        self.module_name = module_name

        self.input_dir = os.getenv("TA1_INPUT_DIR")
        self.map_name = os.getenv("TA1_MAP_NAME")
        self.work_dir = os.getenv("TA1_WORK_DIR")
        self.temp_dir = os.getenv("TA1_TEMP_DIR")
        self.extra_options = os.getenv("TA1_EXTRA_OPTIONS")

    def execute(self) -> int:
        status = self._execute_tool_pre()
        if status:
            return status

        status = self._execute_tool()
        if status:
            return status

        status = self._execute_tool_post()
        if status:
            return status

        return 0

    def _execute_tool(self) -> int:
        completion = subprocess.run(
            args=["python", self.module_name],
            stderr=subprocess.STDOUT)
        print(completion.stdout)
        status = completion.returncode
        if status:
            self._error(f"subprocess failed: {self.module_name} ({status})")
            return status
        return 0

    @abstractmethod
    def _execute_tool_pre(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def _execute_tool_post(self) -> int:
        raise NotImplementedError()

    @staticmethod
    def _error(message: str) -> None:
        print(message)

    def _verify_file_exists(self, path: Path, min_bytes: int = 0) -> None:
        if not path.exists():
            self._error(f"file does not exist: {path}")
            return
        if not path.is_file():
            self._error(f"object is not a file: {path}")
            return
        byte_count = os.path.getsize(path)
        if byte_count < min_bytes:
            self._error(f"file size is too small: {path} (actual={byte_count}, minimum={min_bytes}")
            return

    def _verify_directory_exists(self, path: Path, min_files: int = 0) -> None:
        if not path.exists():
            self._error(f"directory does not exist: {path}")
            return
        if not path.is_dir():
            self._error(f"object is not a directory: {path}")
            return
        file_count = len([i for i in os.listdir('.') if os.path.isfile(i)])
        if file_count < min_files:
            self._error(f"directory file count is too small: {path} (actual={file_count}, minimum={min_files}")
            return
