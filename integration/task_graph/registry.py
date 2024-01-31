# Copyright 2024 InferLink Corporation

from typing import Callable, Optional

_TASKS: dict[str, Callable] = {}


def register(cls):
    _TASKS[cls.NAME] = cls


def registry_lookup(name: str) -> Optional[Callable]:
    return _TASKS.get(name, None)
