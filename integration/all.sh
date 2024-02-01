#!/bin/bash

#set -e

run_main() {
    job_name=$1
    task_name=$2
    output_dir=$3

    echo ""
    echo "*************************************************"
    echo "*** $job_name $task_name ($output_dir)"
    echo "*************************************************"
    echo ""

    rm -fr ~/dev/ta1-jobs/$output_dir/$task_name*

    ./main.py --job-name $job_name --task-name $task_name --job-id $output_dir
}

inputs="WY_CO_Peach WY_EatonRes AK_Dillingham"
modules="map_segment legend_segment map_crop text_spotting legend_item_segment legend_item_description line_extract polygon_extract"

inputs="WY_CO_Peach"
modules="end"

for module in $modules
do
    for input in $inputs
    do
       run_main $input $module 0131_$input
    done
done
