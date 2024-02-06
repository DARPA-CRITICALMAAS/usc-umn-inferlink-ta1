# Map Legend Item Segmentation for Geological Maps

## Environment


### Create from Separate Steps

1. Install pytesseract on the device.

2. Download the mapKurator output geojson(s) from [GoogleDrive](https://docs.google.com/document/d/1mtaGmaNTCy5ybuC0YABqSzoZQAWOCNqHBWbg7moZ0Eo/edit?usp=sharing) if preferred.

3. Make sure you have the output from Uncharted's legend-area segmentation module.

4. Run or manually install the listed libraries in requirement.txt.


### Method

This approach is designed to extract legend items for all point, line, and polygon features from the map-legend area. For extracting polygon legend items, this approach adapts some of UIUC's methods in legend-item segmentation.


## Usage

```
python map_legend_segmentation.py --input_image xxx.tif --output_dir xxx --postprocessing_for_crs True --path_to_mapkurator_output xxx --path_to_intermediate xxx --input_legend_segmentation xxx.json --competition_custom xxx --version 1.2 --log xxx.log
```


```
--input_image: (str) path to the source map tif.
--output_dir: (str) directory to the output (geo)json(s).
--postprocessing_for_crs: (bool) if transforming crs of output geojson is needed.
--path_to_mapkurator_output: (str, optional) path to the mapkurator output geojson.
--path_to_intermediate: (str, optional) directory that stores intermediate outputs. Default to 'intermediate'
--input_legend_segmentation: (str) path to the Uncharted's legend-area segmentation output json.
--competition_custom: (boo, optional) if map-area/ legend-area segmentation output from Uncharted follows their competition schema. Default to False.
--version: (str, optional) a value indicating the version. Default to '1.2'.
--log: (str, optional) path to the logging txt. Default to 'log_file.txt'.
```

Please ignore the error messages regarding indexes and projections if there are any.

