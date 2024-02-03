#!/bin/bash

echo $@

cd /home/mapkurator-system

/root/anaconda3/bin/conda run -n mapkurator \
  python -W ignore /home/mapkurator-system/run_text_spotting.py $@

