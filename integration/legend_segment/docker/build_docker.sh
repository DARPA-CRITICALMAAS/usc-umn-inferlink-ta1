#!/bin/sh
set -e

HERE=`pwd`
cd $REPO_ROOT/../uncharted-ta1/pipelines/segmentation/deploy
./build.sh
cd $HERE

$REPO_ROOT/integration/docker-tools/dockerfile-include.py \
    Dockerfile Dockerfile.tmp $REPO_ROOT/integration/docker-tools

docker build -t inferlink/ta1_legend_segment -f Dockerfile.tmp \
    $REPO_ROOT/../uncharted-ta1

