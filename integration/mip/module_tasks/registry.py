# Copyright 2024 InferLink Corporation

from typing import Optional

from mip.utils.simple_task import SimpleTask


_TASKS: dict[str, type[SimpleTask]] = dict()


def registry_lookup(name: str) -> Optional[type[SimpleTask]]:
    return _TASKS.get(name, None)


# class decorator
def register_task(cls: type[SimpleTask]):
    _TASKS[cls.NAME] = cls
    return cls


def get_tasks() -> dict[str, type[SimpleTask]]:
    return _TASKS
