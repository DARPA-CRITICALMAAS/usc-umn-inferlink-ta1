#!/bin/sh
set -e

$REPO_ROOT/integration/docker-tools/dockerfile-include.py \
    Dockerfile Dockerfile.tmp $REPO_ROOT/integration/docker-tools
docker build \
       -t inferlink/ta1_legend_item_description \
       --build-arg openai_api_key=`cat ~/.ssh/openai` \
       -f Dockerfile.tmp $REPO_ROOT
