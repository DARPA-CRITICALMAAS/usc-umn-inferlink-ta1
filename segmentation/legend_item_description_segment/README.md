# Pairs of Map Key (symbols) and Descriptions Extraction from Geological Maps
## Overview
This module extracts the pairs of map keys (symbols) and corresponding descriptions in the map legend areas from the geological maps.

## Environment
- python 3.10.11
- pytesseract 0.3.10

## Running Command
```sh
python gpt4_main.py --map_dir /dir/of/maps --legend_json_path /dir/of/Uncharted/legend/segment/results --symbol_json_dir /dir/of/USC/legend/segment/results --map_name 'AK_Dillingham' --gpt4_input_dir /dir/of/images --gpt4_output_dir /dir/to/save/json/output
```

Notes: The 'gpt4_input_dir' directory stores cropped images. Given GPT-4's preference for smaller images, this module initially crops the legend areas into smaller areas before feeding them into GPT-4.
