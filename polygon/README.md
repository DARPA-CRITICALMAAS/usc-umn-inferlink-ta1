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
python loam_handler.py --path_to_tif xxx.tif --path_to_json xxx.json --path_to_legend_solution xxx.geojson --path_to_bound xxx.geojson --dir_to_integrated_output xxx
```
if one already has the outputs of legend-item segmentation (for 'path_to_legend_solution') and map-area segmentation (for 'path_to_bound').

or

```
python loam_handler.py --path_to_tif xxx.tif --path_to_json xxx.json --path_to_legend_solution xxx.geojson --map_area_segmentation True --dir_to_integrated_output xxx
```
if one needs map-area segmentation.

Descriptions of the inputs are as follows.

```
--path_to_tif: (str) path to the source map tif.
--path_to_json: (str) path to the source map json. This conforms with the competition json format. I will merge this into 'path_to_legend_solution' once the new gpkg schema/ format is settled and agreed upon by everyone.
--path_to_legend_solution: (str, optional) path to the legend-item segmentation output geojson. Please refer to 'usc-umn-inferlink-ta1/segmentation/legend_item_segmentation/' to run and get the output geojson. If no valid file for this argument is provided, one will only get raster outputs tif in 'LOAM_Intermediate/Metadata_Preprocessing/intermediate7(2)/Output'. This will be in gpkg schema/ format once that discussion is settled.
--path_to_bound: (str, optional) path to the map-area segmentation output geojson. Please refer to 'usc-umn-inferlink-ta1/segmentation/' to run and get the output geojson. This will be in gpkg schema/ format once that discussion is settled.
--dir_to_integrated_output: (str, optional) directory to the vectorization outputs geojson. This will be in gpkg schema/ format once that discussion is settled. (Default to 'Vectorization_Output/')
--dir_to_intermediate: (str, optional) directory to the intermediate outputs in metadata preprocessing. (Default to 'LOAM_Intermediate/Metadata_Preprocessing/')
--dir_to_groundtruth: (str, optional) directory to the groundtruth data tif.
--map_area_segmentation: (bool, optional) whether map-area segmentation is needed. (Default to 'False')
--performance_evaluation: (bool, optional) whether performance evaluation is needed. Please provide 'dir_to_groundtruth' if set to True. (Default to 'False')
```

