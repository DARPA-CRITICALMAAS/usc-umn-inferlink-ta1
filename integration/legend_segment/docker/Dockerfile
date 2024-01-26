FROM docker.uncharted.software/segmentation:latest

# INCLUDEX perms.txt

COPY . /ta1/dev/uncharted-ta1/

WORKDIR /ta1/dev/uncharted-ta1
ENV PYTHONPATH=/ta1/dev/uncharted-ta1:$PYTHONPATH

CMD []
ENTRYPOINT ["python3", "pipelines/segmentation/run_pipeline.py"]
