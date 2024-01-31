#!/usr/bin/env python3
# Copyright 2024 InferLink Corporation

import argparse
import os
from pathlib import Path
from pprint import pprint
import sys
import yaml

import luigi

from resolver import Resolver
from registry import registry_lookup

from simple_task import MyDict
import map_segment_task_1
import legend_segment_task_2
import legend_item_segment_task_3
import legend_item_description_task_4
import map_crop_task_5
import text_spotting_task_6
import line_extract_task_7
import polygon_extract_task_8
import end_task


DEFAULT_JOB_ID = "1"  # datetime.datetime.now().strftime("%H.%M.%S")
DEFAULT_JOB_NAME = "WY_CO_Peach"
DEFAULT_CONFIG_FILE = "./config.yaml"

HOST_ROOT_DIR = "/home/ubuntu/dev"
CONTAINER_ROOT_DIR = "/ta1"


class Options:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--job-id", type=str, default=DEFAULT_JOB_ID)
        parser.add_argument("--job-name", type=str, default=DEFAULT_JOB_NAME)
        parser.add_argument("--config-file", type=str, default=DEFAULT_CONFIG_FILE)
        parser.add_argument("--resolve", action="store_true")
        parser.add_argument("--task-name", type=str, required=True)

        args = parser.parse_args()

        self.job_id = args.job_id
        self.job_name = args.job_name
        self.config_file = args.config_file
        self.resolve = args.resolve
        self.task_name = args.task_name


def main(options: Options) -> int:

    openai_api_key = Path(f"{os.path.expanduser('~')}/.ssh/openai").read_text()

    extras = {
        "JOB_ID": options.job_id,
        "JOB_NAME": options.job_name,
        "CONTAINER_JOB_DIR": CONTAINER_ROOT_DIR + "/job",
        "CONTAINER_DATA_DIR": CONTAINER_ROOT_DIR + "/data",
        "HOST_JOB_DIR": HOST_ROOT_DIR + "/ta1-jobs/" + options.job_id,
        "HOST_DATA_DIR": HOST_ROOT_DIR + "/ta1-data",
        "OPENAI_API_KEY": openai_api_key,
    }
    config_text = Path(options.config_file).read_text()
    data = yaml.load(config_text, Loader=yaml.FullLoader)
    resolver = Resolver(data, extras=extras)
    config_dict = resolver.resolve(data)
    config_dict = MyDict(config_dict)
    if options.resolve:
        pprint(config_dict.data, compact=False, indent=4, width=75)
        return 0

    host_data_dir = Path(config_dict.data["config"]["HOST_DATA_DIR"])
    assert host_data_dir.exists()
    host_job_dir = Path(config_dict.data["config"]["HOST_JOB_DIR"])
    host_job_dir.mkdir(parents=True, exist_ok=True)

    task_cls = registry_lookup(options.task_name)
    if not task_cls:
        print(f"task not found: {options.task_name}")
        return 1

    result = luigi.build(
        tasks=[task_cls(job_id=options.job_id, job_name=options.job_name, _config=config_dict)],
        local_scheduler=True,
        detailed_summary=True
    )

    return 0 if result.status == luigi.execution_summary.LuigiStatusCode.SUCCESS else 1


if __name__ == '__main__':
    opts = Options()
    status = main(opts)
    sys.exit(status)
