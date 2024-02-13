# LOAM

LOAM: Polygon Extraction from Raster Map


## Environment

### Create from Conda Config

```
conda env create -f environment.yml
conda activate loam
```

### Create from Separate Steps
```
conda create -n loam python=3.9.16
conda activate loam

conda install pytorch torchvision==1.13.1 torchaudio==0.13.1 cudatoolkit=11.7 -c pytorch

or

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```


## Usage

### Extracting with a Pre-trained Model

Run the following to exploit the pre-trained polygon-recognition model for extracting polygonal features from raster maps.

```
python loam_handler.py --path_to_tif xxx.tif --path_to_json xxx.json --path_to_legend_solution xxx.geojson --path_to_legend_description xxx.json --dir_to_integrated_output xxx --dir_to_intermediate xxx
```

If one already has the outputs of legend-item segmentation (for 'path_to_json' and 'path_to_legend_solution') and legend-description extraction (for 'path_to_legend_description'), and intends to use the geojson file that conforms with the current schema to identify the legend items. Note that one needs the outputs of legend-item segmentation (for 'path_to_json' and 'path_to_legend_solution') that follow `image coordinate`.

To get the output in image coordinate, one can set 'postprocessing_for_crs' to 'False' to get the output at '(output_dir)/(map_name)/(map_name)_PolygonType_internal.json' and '(output_dir)/(map_name)/(map_name)_PolygonType.geojson' when executing 'usc-umn-inferlink-ta1/segmentation/legend_item_segmentation/'.

Or one can use it in its simplest way:

```
python loam_handler.py --path_to_tif xxx.tif --path_to_json xxx.json --path_to_legend_solution xxx.geojson
```

The outputs will be stored in 'Vectorization_Output/' folder.


Descriptions of the inputs are as follows.

```
--path_to_tif: (str) path to the source map tif.
--path_to_json: (str) path to the source map json. This conforms with the competition json format.
--path_to_legend_solution: (str, optional) path to the legend-item segmentation output geojson. Please refer to 'usc-umn-inferlink-ta1/segmentation/legend_item_segmentation/' to run and get the output geojson. If no valid file for this argument is provided, one will only get raster outputs tif in 'LOAM_Intermediate/Metadata_Preprocessing/intermediate7(2)/Output'.
--path_to_legend_description: (str, optional) path to the legend-description extraction output json.
--path_to_bound: (str, optional) path to the map-area segmentation output geojson. Please refer to 'usc-umn-inferlink-ta1/segmentation/' to run and get the output geojson.
--dir_to_integrated_output: (str, optional) directory to the vectorization outputs geojson. (Default to 'Vectorization_Output/')
--dir_to_intermediate: (str, optional) directory to the intermediate output files. (Default to 'Example_Output/')
--dir_to_groundtruth: (str, optional) directory to the groundtruth data tif.
--set_json: (bool, optional) whether to use the json file that conforms with the competition schema to identify the legend items. (Default to 'True')
--map_area_segmentation: (bool, optional) whether map-area segmentation is needed. (Default to 'False')
--performance_evaluation: (bool, optional) whether performance evaluation is needed. Please provide 'dir_to_groundtruth' if set to True. (Default to 'False')
--version: (str, optional) a value indicating the version. (Default to '0')
--log: (str, optional) path to the logging txt. (Default to 'log_file.txt')
```

