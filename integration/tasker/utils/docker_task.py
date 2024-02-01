# Copyright 2024 InferLink Corporation

from tasker.utils.docker_runner import DockerRunner
from tasker.utils.simple_task import SimpleTask


class DockerTask(SimpleTask):

    NAME = "invalid"
    GPU = True
    USER = "cmaas"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run_body(self):

        container = self._make_container()

        status, log, elapsed = container.run()

        print("-----------------------------------------------")
        print(log)
        print("")
        print(f"status: {status}")
        print(f"elapsed: {elapsed} secs")
        print("-----------------------------------------------")

        if status:
            raise Exception(f"docker run failed: {self.NAME}")

    def _make_container(self) -> DockerRunner:
        image_name = f"inferlink/ta1_{self.NAME}"
        gpus = self.GPU
        docker_log = self.task_config.host_output_dir / f"{self.NAME}.docker.txt"

        environment = [
            f"OPENAI_API_KEY={self.config.openai_key}"
        ]

        user = self.USER

        volumes = [
            f"{self.task_config.host_input_dir}:{self.task_config.container_input_dir}",
            f"{self.task_config.host_output_dir}:{self.task_config.container_output_dir}",
            f"{self.task_config.host_temp_dir}:{self.task_config.container_temp_dir}",
        ]

        options = self.task_config.get_options()

        container = DockerRunner(
            image=image_name,
            name=f"{self.job_name}__{self.NAME}",
            command=options,
            volumes=volumes,
            environment=environment,
            log_file=docker_log,
            user=user,
            gpus=gpus,
        )

        return container
