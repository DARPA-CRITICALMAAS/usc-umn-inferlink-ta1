#!/usr/bin/env python3
# Copyright 2024 InferLink Corporation

import datetime
import os
from pathlib import Path
from pprint import pprint
import shutil
import sys
import yaml

import luigi

from ta1_tasks import TA1Task, MyDict
from resolver import Resolver


DEFAULT_JOB_ID = "1"  # datetime.datetime.now().strftime("%H.%M.%S")
DEFAULT_JOB_NAME = "AK_Dillingham"
#DEFAULT_JOB_NAME = "WY_CO_Peach"
DEFAULT_CONFIG_FILE = "./config.yaml"

HOST_ROOT_DIR = "/home/ubuntu/dev"
CONTAINER_ROOT_DIR = "/ta1"


def parse_args(args: list[str]) -> dict[str, str]:
    job_id = DEFAULT_JOB_ID
    job_name = DEFAULT_JOB_NAME
    config_file = DEFAULT_CONFIG_FILE
    resolve = False

    i = 0
    while i < len(args):
        if args[i] == "--job-id":
            job_id = args[i + 1]
            i += 2
        elif args[i] == "--job-name":
            job_name = args[i + 1]
            i += 2
        elif args[i] == "--config-file":
            config_file = args[i + 1]
            i += 2
        elif args[i] == "--resolve":
            resolve = True
            i += 1
        else:
            print(f"usage error: invalid argument: {args[i]}")
            sys.exit(1)

    if not job_id:
        print("usage error: --job-id missing")
        sys.exit(1)
    if not job_name:
        print("usage error: --job-name missing")
        sys.exit(1)
    if not config_file:
        print("usage error: --config-file missing")
        sys.exit(1)

    return {
        "job_id": job_id, "job_name": job_name,
        "config_file": config_file,
        "resolve": resolve,
    }


def main(
        job_id: str, job_name: str,
        config_file: str,
        resolve: bool) -> int:

    openai_api_key = Path(f"{os.path.expanduser('~')}/.ssh/openai").read_text()

    extras = {
        "JOB_ID": job_id,
        "JOB_NAME": job_name,
        "CONTAINER_JOB_DIR": CONTAINER_ROOT_DIR + "/job",
        "CONTAINER_DATA_DIR": CONTAINER_ROOT_DIR + "/data",
        "HOST_JOB_DIR": HOST_ROOT_DIR + "/ta1-jobs/" + job_id,
        "HOST_DATA_DIR": HOST_ROOT_DIR + "/ta1-data",
        "OPENAI_API_KEY": openai_api_key,
    }
    config_text = Path(config_file).read_text()
    data = yaml.load(config_text, Loader=yaml.FullLoader)
    resolver = Resolver(data, extras=extras)
    config_dict = resolver.resolve(data)
    config_dict = MyDict(config_dict)
    if resolve:
        pprint(config_dict.data, compact=False, indent=4, width=75)
        return 0

    host_data_dir = Path(config_dict.data["config"]["HOST_DATA_DIR"])
    assert host_data_dir.exists()
    host_job_dir = Path(config_dict.data["config"]["HOST_JOB_DIR"])
    #if host_job_dir.exists():
    #    shutil.rmtree(host_job_dir)
    host_job_dir.mkdir(parents=True, exist_ok=True)

    result = luigi.build(
        tasks=[TA1Task(job_id=job_id, job_name=job_name, _config=config_dict)],
        local_scheduler=True,
        detailed_summary=True
    )

    return 0 if result.status == luigi.execution_summary.LuigiStatusCode.SUCCESS else 1


if __name__ == '__main__':
    arguments = parse_args(sys.argv[1:])
    status = main(**arguments)
    sys.exit(status)
