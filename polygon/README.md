# PERMIAN

PERMIAN: Prototype for v2 Polygon Extraction Module


## Environment

This module was tested on a machine equipped with an AMD Ryzen 9 5900X 12-core processor at 3.7 GHz, 128 GB DDR4 RAM (32 GB x 4) at 3200 MHz, with 4 TB SSD. This module does not require an additional database. It exploits multi-processing to achieve better efficiency. To ensure a good user experience, please make sure the tested machine has at least 64 GB RAM (a tested feasible setup) and a processor with more than 10 cores available.


### Create from Conda Config

```
conda env create -f environment.yml
conda activate loam
```

### Create from Separate Steps

```
conda create -n loam python=3.9.16
conda activate loam
pip install -r requirements.txt
```


## Usage

### Extracting wi

Please use it in its simplest way:

```
python permian_handler.py --path_to_tif xxx.tif --path_to_json xxx.json --path_to_legend_solution xxx.geojson --path_to_legend_description xxx.json
```

in this case, the output file will be stored in 'Example_Output/PERMIAN_Vector_Output' folder for vector geojson, and 'Example_Output/PERMIAN_Raster_Output' folder for raster tif.

Or use it with a bit more directory setup for output and intermediate files:

```
python permian_handler.py --path_to_tif xxx.tif --path_to_json xxx.json --path_to_legend_solution xxx.geojson --path_to_legend_description xxx.json --dir_to_vector_output xxx --dir_to_raster_output xxx --dir_to_intermediate xxx
```


Descriptions of the inputs are as follows.

```
--path_to_tif: (str, mandatory) path to the source map tif.
--path_to_json: (str, mandatory) (Legend-item segmentation output) path to the source map json. Please refer to 'usc-umn-inferlink-ta1/segmentation/legend_item_segmentation/' to run and get the output json ([xxx]_PolygonType_internal.json). This conforms with the competition json format.
--path_to_legend_solution: (str, mandatory) (Legend-item segmentation output) path to the legend-item segmentation output geojson. Please refer to 'usc-umn-inferlink-ta1/segmentation/legend_item_segmentation/' to run and get the output geojson ([xxx]_PolygonType.geojson).
--path_to_legend_description: (str, mandatory) (Legend-description segmentation output) path to the legend-description extraction output json. Please refer to 'usc-umn-inferlink-ta1/segmentation/legend_item_description_segment/' to run and get the output json ([xxx]_Polygon.json).

--dir_to_vector_output: (str, optional but recommended) directory to the vectorization outputs geojson. (Default to 'Vectorization_Output/PERMIAN_Vector_Output')
--dir_to_raster_output: (str, optional but recommended) directory to the raster outputs tif. (Default to 'Vectorization_Output/PERMIAN_Raster_Output')
--dir_to_intermediate: (str, optional) directory to the intermediate output files. (Default to 'Example_Output/')
--log: (str, optional) path to the logging file. (Default to 'Example_Output/log.log')

--path_to_bound: (str, optional) path to the map-area segmentation output geojson.
--version: (str, optional) a value indicating the version. (Default to '2')
```
