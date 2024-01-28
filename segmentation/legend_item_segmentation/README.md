# Map Legend Item Segmentation for Geological Maps

## Environment


### Create from Separate Steps

1. Install pytesseract on the device.

2. Download the mapKurator output geojson(s) from [GoogleDrive](https://docs.google.com/document/d/1mtaGmaNTCy5ybuC0YABqSzoZQAWOCNqHBWbg7moZ0Eo/edit?usp=sharing).

3. Download the pre-train model from [GoogleDrive](https://drive.google.com/drive/folders/19YRDJYKhSb3DzTLFKLNSAoKFfURL9FgH?usp=sharing).

3. Run or manually install the listed libraries in requirement.txt.
```
pip install -r requirement.txt
```



## Usage

If one has a tif for map-area segmentation output:

```
python map_legend_segmentation.py --input_image xxx.tif --output_dir xxx --preprocessing_for_cropping True --postprocessing_for_crs True --path_to_mapkurator_output xxx --path_to_intermediate xxx --input_area_segmentation xxx.tif --input_legend_segmentation xxx.tif --competition_custom xxx --version 1 --log xxx.txt
```

If one has a json for map-area segmentation output, following Uncharted's schema:

```
python map_legend_segmentation.py --input_image xxx.tif --output_dir xxx --preprocessing_for_cropping True --postprocessing_for_crs True --path_to_mapkurator_output xxx --path_to_intermediate xxx --input_area_segmentation xxx.json --input_legend_segmentation xxx.tif --competition_custom xxx --version 1 --log xxx.txt
```

```
--input_image: (str) path to the source map tif.
--output_dir: (str) directory to the output json(s).
--preprocessing_for_cropping: (bool) if map area segmentation is needed.
--postprocessing_for_crs: (bool) if transforming crs of output json is needed.
--path_to_mapkurator_output: (str) path to the mapkurator output geojson.
--path_to_intermediate: (str) directory that stores intermediate outputs. Default to 'intermediate'
--input_area_segmentation: (str, optional) path to the map area segmentation output tif.
--input_legend_segmentation: (str, optional) path to the map legend segmentation output. There are 2 formats for this input: (1) A tif that is a binary mask highlighting polygon/line/point legend areas. (2) A json that follows the Uncharted map-area segmentation output schema.
--competition_custom: (boo, optional) if map-area/ legend-area segmentation output from Uncharted follows their competition schema. Default to False.
--version: (str, optional) a value indicating the version. Default to '0'.
--log: (str, optional) path to the logging txt. Default to 'log_file.txt'.
```

Please ignore the error messages regarding indexes and projections if there are any.

<del> The time spent for this approach depends on the complexity (i.e., the number of legend items, the amount of texts, and the layouts) of the map. </del>
