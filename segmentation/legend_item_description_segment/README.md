# Pairs of Map Key (symbols) and Descriptions Extraction from Geological Maps
## Overview
This module extracts the pairs of map keys (symbols) and corresponding descriptions in the map legend areas from the geological maps.

## GPT4 API
Before proceeding, please ensure that you have properly configured the GPT-4 API, as this module relies on its functionality. Make sure that your GPT-4 API setup is in place and ready for use.

## Environment
- python 3.10.11
- pytesseract 0.3.10

## Running Command
```sh
python gpt4_main.py --map_dir /dir/of/maps --legend_json_dir /dir/of/Uncharted/legend/segment/results --symbol_json_dir /dir/of/USC/legend/segment/results --map_name 'AK_Dillingham' --temp_dir /dir/of/images --output_dir /dir/to/save/json/output --log_path /path/to/save/log
```
