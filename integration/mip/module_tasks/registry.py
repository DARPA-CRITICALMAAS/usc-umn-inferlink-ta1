# Copyright 2024 InferLink Corporation

from typing import Callable, Optional

_TASKS: dict[str, Callable] = dict()


def registry_lookup(name: str) -> Optional[Callable]:
    return _TASKS.get(name, None)


# class decorator
def register_task(cls):
    _TASKS[cls.NAME] = cls
    return cls


def get_task_names() -> list[str]:
    return list(_TASKS.keys())
