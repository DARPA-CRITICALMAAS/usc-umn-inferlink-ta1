# Map Legend Item Segmentation for Geological Maps

## Environment


### Create from Separate Steps

1. Install pytesseract on the device.

2. Download the mapKurator output geojson(s) from [GoogleDrive](https://docs.google.com/document/d/1mtaGmaNTCy5ybuC0YABqSzoZQAWOCNqHBWbg7moZ0Eo/edit?usp=sharing).

3. Run or manually install the listed libraries in requirement.txt.
```
pip install -r requirement.txt
```



## Usage

```
python map_legend_segmentation.py --input_image xxx.tif --output_dir xxx --preprocessing_for_cropping True --postprocessing_for_crs True --path_to_mapkurator_output xxx --path_to_intermediate xxx --input_area_segmentation xxx.tif --input_legend_segmentation xxx.tif
```

```
--input_image: (str) path to the source map tif.
--output_dir: (str) directory to the output json(s).
--preprocessing_for_cropping: (bool) if map area segmentation is needed.
--postprocessing_for_crs: (bool) if transforming crs of output json is needed.
--path_to_mapkurator_output: (str) path to the mapkurator output geojson.
--path_to_intermediate: (str) directory that stores intermediate outputs. This is default to 'intermediate'
--input_area_segmentation: (str, optional) path to the map area segmentation output tif.
--input_legend_segmentation: (str, optional) path to the map legend segmentation output tif. This tif shall be a binary mask that highlights polygon/line/point legend areas.
```

Please ignore the error messages regarding indexes in Step (2/9) if there are any.

The time spent for this approach depends on the complexity (i.e., the number of legend items, the amount of texts, and the layouts) of the map.
