#!/bin/sh
set -e

$REPO_ROOT/integration/docker-tools/dockerfile-include.py \
    Dockerfile Dockerfile.tmp $REPO_ROOT/integration/docker-tools

docker build -t inferlink/ta1_map_segment -f Dockerfile.tmp $REPO_ROOT
