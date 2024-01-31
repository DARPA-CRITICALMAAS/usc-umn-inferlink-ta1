# Copyright 2024 InferLink Corporation

import os
from pathlib import Path
import shutil
from typing import Optional

import luigi

from docker_runner import DockerRunner
from simple_task import SimpleTask


class DockerTask(SimpleTask):

    NAME = "invalid"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        tool_options = self.config.data[self.NAME]["options"]
        tool_config = self.config.data[self.NAME]["config"]

        dirs = self.config.data[self.NAME]["config"].get("make_dirs", [])
        for d in dirs:
            if not Path(d).exists():
                Path(d).mkdir(parents=True, exist_ok=True)
        #     shutil.rmtree(d, ignore_errors=True)
        #     os.makedirs(d)

        image = tool_config["docker_image"]

        gpus = tool_config.get("gpus", "True").lower().strip() in ("yes", "y", "true", "t", "1")

        docker_log = Path(f"{self.config.data['config']['HOST_JOB_DIR']}/{self.NAME}.docker.log")

        options = []
        if tool_options:
            for k, v in tool_options.items():
                options.append("--" + k)
                if type(v) is str:
                    options.append(v)
                elif type(v) is list:
                    for vv in v:
                        options.append(str(vv))
                else:
                    raise Exception(f"unhandled option type {type(v)}: --{k} {v}")

        volumes = [
            f"{self.host_job_dir}:{self.container_job_dir}",
            f"{self.host_data_dir}:{self.container_data_dir}"
        ]

        environment = [
            f"OPENAI_API_KEY={self.config.data['config']['OPENAI_API_KEY']}"
        ]

        user = tool_config.get("user", "cmaas")

        self._docker = DockerRunner(
            image=image,
            name=self.NAME,
            command=options,
            volumes=volumes,
            environment=environment,
            log_file=docker_log,
            user=user,
            gpus=gpus,
        )

    def run(self):
        status, log, elapsed = self._docker.run()

        if status != 0:
            print("-----------------------------------------------")
            print(log)
            print("-----------------------------------------------")
            raise Exception(f"docker run failed: {self.NAME}")

        output_errs = self._check_outputs_exist()

        with self.output().open('w') as f:
            f.write(f"{self.job_id} {self.job_name}\n")
            f.write("----------------------------------------------------\n")
            f.write(log)
            f.write("----------------------------------------------------\n")

            if output_errs:
                for output_err in output_errs:
                    f.write(f"output missing: {output_err}\n")
                f.write("----------------------------------------------------\n")

            f.write(f"elapsed: {elapsed} secs\n")
            f.write(f"status: {status}\n")

        if output_errs:
            raise Exception(f"file/dir check failed: {self.NAME}")

    def _get_expected_outputs(self) -> tuple[list[Path], list[Path]]:
        raise NotImplementedError()

    def _check_outputs_exist(self) -> list[str]:
        errs: list[str] = list()

        expected_files, expected_dirs = self._get_expected_outputs()
        for file in expected_files:
            err = self._check_file_exists(file)
            if err:
                errs.append(err)
        for dr in expected_dirs:
            err = self._check_dir_exists(dr)
            if err:
                errs.append(err)

        return errs

    def _check_file_exists(self, file: Path) -> Optional[str]:
        file = Path(self.config.data["config"]["HOST_JOB_DIR"], self.NAME, "output", file)
        if not file.exists():
            return f"file not found: {file}"
        if not file.is_file():
            return f"file is not a file: {file}"
        if os.path.getsize(file) == 0:
            return f"file has zero length: {file}"
        return None

    def _check_dir_exists(self, dr: Path) -> Optional[str]:
        dr = Path(self.config.data["config"]["HOST_JOB_DIR"], self.NAME, "output", dr)
        if not dr.exists():
            return f"dir not found: {dr}"
        if not dr.is_dir():
            return f"dir is not a dir: {dr}"
        count = len([i for i in os.listdir(dr)])
        if count == 1: # just the dir itself
            return f"dir has zero files: {dr}"
        return None

    def output(self):
        return luigi.LocalTarget(f"{self.host_job_dir}/{self.NAME}.txt")
