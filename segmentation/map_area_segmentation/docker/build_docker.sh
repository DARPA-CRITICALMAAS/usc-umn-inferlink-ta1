#!/bin/sh
set -e

docker pull pytorch/pytorch

$REPO_ROOT/integration/docker-tools/dockerfile-include.py \
    Dockerfile Dockerfile.tmp $REPO_ROOT/integration/docker-tools

docker build -t inferlink/ta1_map_segment -f Dockerfile.tmp $REPO_ROOT

docker push inferlink/ta1_map_segment

