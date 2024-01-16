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
pip install -r requirements.txt
```


## Usage

### Extracting with a Pre-trained Model

1. We provide the link to our pre-trained model used for inference in `\LOAM\checkpoints\checkpoint_epoch2.pth_link.txt`, or access here to [Google Drive](https://drive.google.com/file/d/16N-2NbtqLSNCU83J5Iyw4Ca8A2f7aIqi/view?usp=sharing).

2. Run the following to exploit the pre-trained polygon-recognition model for extracting polygonal features from raster maps.

```
python loam_handler.py --path_to_tif xxx.tif --path_to_legend_solution xxx.geojson --path_to_bound xxx.geojson --dir_to_integrated_output xxx
```

If one already has the outputs of legend-item segmentation (for 'path_to_legend_solution') and map-area segmentation (for 'path_to_bound'), and intends to use the geojson file that conforms with the current schema to identify the legend items. Note that one needs the outputs of legend-item segmentation (for 'path_to_legend_solution') that follows `image coordinate`.

To get the output in image coordinate, one can set 'postprocessing_for_crs' to 'False' to get the output at '(output_dir)/(map_name)/(map_name)_PolygonType.geojson' when executing 'usc-umn-inferlink-ta1/segmentation/legend_item_segmentation/'; or one can get the output from '(path_to_intermediate)/intermediate7/(map_name)_PolygonType.geojson' when setting 'postprocessing_for_crs' to 'True'.

Or one can use it in its simplest way:

```
python loam_handler.py --path_to_tif xxx.tif --path_to_legend_solution xxx.geojson
```

The outputs will be stored in 'Vectorization_Output/' folder.


3. Here are some alternative ways to run the process.

(1) If one needs map-area segmentation:

```
python loam_handler.py --path_to_tif xxx.tif --path_to_legend_solution xxx.geojson --map_area_segmentation True --dir_to_integrated_output xxx
```

(2) If one intends to use the json file that conforms with the competition schema to identify the legend items:

```
python loam_handler.py --path_to_tif xxx.tif --path_to_json xxx.json --path_to_legend_solution xxx.geojson --path_to_bound xxx.geojson --dir_to_integrated_output xxx --set_json False
```

(3) If one does not have legend-item segmentation output and only needs polygon-extraction outputs in raster format (There will be no vector outputs if --path_to_legend_solution is empty):

```
python loam_handler.py --path_to_tif xxx.tif --path_to_json xxx.json --dir_to_integrated_output xxx
```

Descriptions of the inputs are as follows.

```
--path_to_tif: (str) path to the source map tif.
--path_to_json: (str) path to the source map json. This conforms with the competition json format. I will merge this into 'path_to_legend_solution' once the new gpkg schema/ format is settled and agreed upon by everyone.
--path_to_legend_solution: (str, optional) path to the legend-item segmentation output geojson. Please refer to 'usc-umn-inferlink-ta1/segmentation/legend_item_segmentation/' to run and get the output geojson. If no valid file for this argument is provided, one will only get raster outputs tif in 'LOAM_Intermediate/Metadata_Preprocessing/intermediate7(2)/Output'. This will be in gpkg schema/ format once that discussion is settled.
--path_to_bound: (str, optional) path to the map-area segmentation output geojson. Please refer to 'usc-umn-inferlink-ta1/segmentation/' to run and get the output geojson. This will be in gpkg schema/ format once that discussion is settled.
--dir_to_integrated_output: (str, optional) directory to the vectorization outputs geojson. This will be in gpkg schema/ format once that discussion is settled. (Default to 'Vectorization_Output/')
--dir_to_groundtruth: (str, optional) directory to the groundtruth data tif.
--set_json: (bool, optional) whether to use the json file that conforms with the competition schema to identify the legend items. (Default to 'False')
--map_area_segmentation: (bool, optional) whether map-area segmentation is needed. (Default to 'False')
--performance_evaluation: (bool, optional) whether performance evaluation is needed. Please provide 'dir_to_groundtruth' if set to True. (Default to 'False')
--version: (str, optional) a value indicating the version. (Default to '0')
--log: (str, optional) path to the logging txt. (Default to 'log_file.txt')
```

