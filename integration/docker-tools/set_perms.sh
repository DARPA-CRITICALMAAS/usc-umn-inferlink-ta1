#!/bin/sh

set -e

DIR1=/home/ubuntu/dev/ta1-data
DIR2=/home/ubuntu/dev/ta1-jobs

sudo groupdel 1024 || true
sudo groupdel cmaasgroup || true
sudo addgroup --gid 1024 cmaasgroup
sudo adduser ubuntu cmaasgroup

# make the mounted volume accessible
sudo chown -R :cmaasgroup $DIR1 $DIR2
sudo chmod -R ug=rwX $DIR1 $DIR2
sudo chmod -R g+s $DIR1 $DIR2

echo Done!
