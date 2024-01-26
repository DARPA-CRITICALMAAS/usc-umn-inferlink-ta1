#!/bin/bash
set -e

export MAP_SEGMENT=segmentation/map_area_segmentation
export LEGEND_SEGMENT=integration/legend_segment
export LEGEND_ITEM_SEGMENT=segmentation/legend_item_segmentation
export LEGEND_ITEM_DESCRIPTION=segmentation/legend_item_description_segment
export MAP_CROP=system/image_crop
export TEXT_SPOTTING=integration/text_spotting
export LINE_EXTRACT=line
export POLYGON_EXTRACT=polygon

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
