# Copyright 2024 InferLink Corporation

from pathlib import Path
import os


def _error(message: str):
    raise Exception(message)


def check_file_exists(path: str | Path, min_bytes: int = 0) -> None:
    path = Path(path) if type(path) is str else path

    if not path.exists():
        _error(f"file does not exist: {path}")
        return
    if not path.is_file():
        _error(f"object is not a file: {path}")
        return
    byte_count = os.path.getsize(path)
    if byte_count < min_bytes:
        _error(f"file size is too small: {path} (actual={byte_count}, minimum={min_bytes}")
        return


def check_directory_exists(path: str | Path, min_files: int = 0) -> None:
    path = Path(path) if type(path) is str else path

    if not path.exists():
        _error(f"directory does not exist: {path}")
        return
    if not path.is_dir():
        _error(f"object is not a directory: {path}")
        return
    file_count = len([i for i in os.listdir('.') if os.path.isfile(i)])
    if file_count < min_files:
        _error(f"directory file count is too small: {path} (actual={file_count}, minimum={min_files}")
        return
