#!/bin/bash
set -e

do_pull=0
do_build=0
do_push=0

while (($#))
do
    case $1 in
	--pull) do_pull=1; shift;;
	--build) do_build=1; shift;;
	--push) do_push=1; shift;;
	*) echo "usage: build.sh [--pull] [--build] [--push]" ; false 
    esac
done


. ./modules.sh

if [ "$do_pull" -eq 1 ]
then
    for i in $MODULE_IMAGES
    do
	echo ""
	echo "*** $i... ***"
	echo ""
	docker pull $i
    done
fi

if [ "$do_build" -eq 1 ]
then
    for i in $MODULE_DIRS
    do
	echo ""
	echo "*** $i... ***"
	echo ""
	pushd $REPO_ROOT/$i
	./build_docker.sh
	popd
    done
fi

if [ "$do_push" -eq 1 ]
then
    for i in $MODULE_IMAGES
    do
	echo ""
	echo "*** $i... ***"
	echo ""
	docker push $i
    done
fi
