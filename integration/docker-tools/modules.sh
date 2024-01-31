#!/bin/bash
set -e

export MAP_SEGMENT=integration/1_map_segment
export LEGEND_SEGMENT=integration/2_legend_segment
export LEGEND_ITEM_SEGMENT=integration/3_legend_item_segment
export LEGEND_ITEM_DESCRIPTION=integration/4_legend_item_description
export MAP_CROP=integration/5_map_crop
export TEXT_SPOTTING=integration/6_text_spotting
export LINE_EXTRACT=integration/7_line_extract
export POLYGON_EXTRACT=integration/8_polygon_extract

export MODULE_DIRS=" \
    $MAP_SEGMENT \
    $LEGEND_SEGMENT \
    $LEGEND_ITEM_SEGMENT \
    $LEGEND_ITEM_DESCRIPTION \
    $MAP_CROP \
    $TEXT_SPOTTING \
    $LINE_EXTRACT \
    $POLYGON_EXTRACT \
"

export MODULE_IMAGES=" \
    inferlink/ta1_map_segment \
    inferlink/ta1_legend_segment \
    inferlink/ta1_legend_item_segment \
    inferlink/ta1_legend_item_description \
    inferlink/ta1_map_crop \
    inferlink/ta1_text_spotting \
    inferlink/ta1_line_extract \
    inferlink/ta1_polygon_extract \
"
