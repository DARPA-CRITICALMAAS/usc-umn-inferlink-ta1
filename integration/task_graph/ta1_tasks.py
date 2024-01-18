# Copyright 2024 InferLink Corporation

import os
from pathlib import Path
from typing import cast

import luigi
from docker_runner import DockerRunner


class MyDict:
    def __init__(self, d):
        self.data = d


class SimpleTask(luigi.Task):
    job_id = luigi.Parameter()
    job_name = luigi.Parameter()
    _config = luigi.Parameter(significant=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.container_data_dir = self.config.data["config"]["DATA_DIR"]
        self.container_job_dir = self.config.data["config"]["JOB_DIR"]
        self.host_data_dir = self.config.data["config"]["HOST_DATA_DIR"]
        self.host_job_dir = self.config.data["config"]["HOST_JOB_DIR"]

    @property
    def config(self) -> MyDict:
        return cast(MyDict, self._config)


class DockerTask(SimpleTask):

    NAME = "invalid"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        tool_options = self.config.data[self.NAME]["options"]
        tool_config = self.config.data[self.NAME]["config"]

        dirs = self.config.data[self.NAME]["config"].get("make_dirs", [])
        for d in dirs:
            os.makedirs(d, exist_ok=True)

        image = tool_config["docker_image"]

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
        )

    def run(self):
        status, log = self._docker.run()
        if status != 0:
            print("-----------------------------------------------")
            print(log)
            print("-----------------------------------------------")
            raise Exception(f"docker run failed: {self.NAME}")

        with self.output().open('w') as f:
            f.write(f"{self.job_id} {self.job_name}\n")
            f.write(str(self.config.data[self.NAME]))

    def output(self):
        return luigi.LocalTarget(f"{self.host_job_dir}/{self.NAME}-task.txt")


class StartTask(SimpleTask):

    NAME = "start"

    def output(self):
        return luigi.LocalTarget(f"{self.host_job_dir}/{self.NAME}-task.txt")

    def run(self):
        with self.output().open('w') as f:
            f.write(f"{self.job_id} {self.job_name}")


class MapSegmentTask(DockerTask):

    NAME = "map_segment"

    def requires(self):
        return StartTask(job_id=self.job_id, job_name=self.job_name, _config=self.config)


class LegendSegmentTask(DockerTask):

    NAME = "legend_segment"

    def requires(self):
        return StartTask(job_id=self.job_id, job_name=self.job_name, _config=self.config)


class LegendItemSegmentTask(DockerTask):

    NAME = "legend_item_segment"

    def requires(self):
        return StartTask(job_id=self.job_id, job_name=self.job_name, _config=self.config)


class LegendItemDescriptionTask(DockerTask):

    NAME = "legend_item_description"

    def requires(self):
        return StartTask(job_id=self.job_id, job_name=self.job_name, _config=self.config)


class MapCropTask(DockerTask):

    NAME = "map_crop"

    def requires(self):
        return StartTask(job_id=self.job_id, job_name=self.job_name, _config=self.config)


class TextSpottingTask(DockerTask):

    NAME = "text_spotting"

    def requires(self):
        return StartTask(job_id=self.job_id, job_name=self.job_name, _config=self.config)


class LineExtractTask(DockerTask):

    NAME = "line_extract"

    def requires(self):
        return StartTask(job_id=self.job_id, job_name=self.job_name, _config=self.config)


class PolygonExtractTask(DockerTask):

    NAME = "polygon_extract"

    def requires(self):
        return StartTask(job_id=self.job_id, job_name=self.job_name, _config=self.config)


class TA1Task(SimpleTask):

    NAME = "ta1"

    def requires(self):
        return [
            StartTask(job_id=self.job_id, job_name=self.job_name, _config=self.config),

            # 1
            # MapSegmentTask(job_id=self.job_id, job_name=self.job_name, _config=self.config),

            # 2- uncharted
            # LegendSegmentTask(job_id=self.job_id, job_name=self.job_name, _config=self.config),

            # 3
            # LegendItemSegmentTask(job_id=self.job_id, job_name=self.job_name, _config=self.config),

            # 4
            # LegendItemDescriptionTask(job_id=self.job_id, job_name=self.job_name, _config=self.config),

            # 5
            MapCropTask(job_id=self.job_id, job_name=self.job_name, _config=self.config),

            # 6
            # TextSpottingTask(job_id=self.job_id, job_name=self.job_name, _config=self.config),

            # 7
            # LineExtractTask(job_id=self.job_id, job_name=self.job_name, _config=self.config),

            # 8
            # PolygonExtractTask(job_id=self.job_id, job_name=self.job_name, _config=self.config),
        ]

    def output(self):
        return luigi.LocalTarget(f"{self.host_job_dir}/{self.NAME}-task.txt")

    def run(self):
        with self.output().open('w') as f:
            f.write(f"{self.job_id} {self.job_name}")
