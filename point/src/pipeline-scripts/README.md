# Point Feature Pipleline

## Dependencies:

This module is tested with (1) hardware NVIDIA A100 GPU, (2) CUDA 11.3 and (3) Pytorch 1.10.0.
Below are steps for conda environment settings 

### Setting with requirements.txt
```
conda create -n pnt_pipe python=3.8 -y
conda activate pnt_pipe
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt

```

### Setting with conda environment.yaml
```
conda env create -f point_environment.yml
conda activate pnt_pipe
```

## Instruction:

To run the point symbol pipeline, use the following command:

(1) Generate raster layer output: (add ```--save_raster``` argument ) 

``` 
python run_point_pipe.py --map_dir /your/map/directory --map_metadata_dir /your/metadata/directory --map_patches_dir /your/patches/directory/per/map --model_weights_dir model_weight/ --output_dir_root /your/output/root/directory --symbol_info_json_file automated_model_selection/symbol_info.json --save_raster --gpu_id 0
```
(2) Generate geojson output: (remove ```--save_raster``` argument ) 

``` 
python run_point_pipe.py --map_dir /your/map/directory --map_metadata_dir /your/metadata/directory --map_patches_dir /your/patches/directory/per/map --model_weights_dir model_weight/ --output_dir_root /your/output/root/directory --symbol_info_json_file automated_model_selection/symbol_info.json --gpu_id 0 
```

(3) Evaluate with the competition evaluation dataset : 

(add ```--cmp_eval_gt_path automated_model_selection/cmp-eval-pair.json --save_raster --cmp_eval ``` argument ) 

We support generating raster outputs from the competition evaluation dataset. Please follow the command line below to generate raster outputs:
``` 
python run_point_pipe.py --map_dir /your/map/directory --model_weights_dir model_weight/ --map_patches_dir /your/patches/directory/per/map --output_dir_root /your/output/root/directory --symbol_info_json_file automated_model_selection/symbol_info.json --cmp_eval_gt_path automated_model_selection/cmp-eval-pair.json --save_raster --cmp_eval --gpu_id 0
```

This repository contains (1) pretrained models dir  (```--model_weights_dir model_weight/``` ) ,  (2) a metadata used for selecting pretrained models per maps (```--symbol_info_json_file automated_model_selection/symbol_info.json```) and (3) point symbol pairs on competition evaluation data (```--cmp_eval_gt_path automated_model_selection/cmp-eval-pair.json``` )


## Command Line Arguments:

```
--map_dir : Directory containing entire map sheets
--map_metadata_dir: Directory to map metadata that contains the information about coordinate of legend and map dimension. 
--map_patches_dir: Directory for input map patches directory per single map 
--model_weights_dir: Directory to pretrained point feature detection model weights  
--symbol_info_json_file : Json file used for selecting pretrained models based on text description on a map legend 
--output_dir_root: Root directory for output directory
--save_raster : Enable this argument if you want to generate output with raster layer
--cmp_eval_gt_path : Json file contains point symbol pairs in the competition evaluation dataset
--cmp_eval : Enable this argument if you want to generate competition evaluation outputs
--gpu_id : Specifying GPU id for running the module
--text_spotting_dir (Optional) : Directory of mapKurator outputs. This is optional, which is used for postprocessing point symbol module outputs

```

<!-- --cropped_legend_dir: Directory to save the cropped legend <br>
--template_dir: Directory to the legend template  <br>
--processed_legend_dir: Directory to save the processed cropped legend  <br> -->

Feel free to replace "/your/map/directory" and other placeholders with the 
actual paths relevant to your project
 
