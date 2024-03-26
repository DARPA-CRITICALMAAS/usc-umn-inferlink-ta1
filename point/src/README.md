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

``` python run_point_pipe.py --map_dir /your/map/directory --map_metadata_dir /your/metadata/directory --map_patches_dir /your/patches/directory --model_weights_dir ./pipeline-scripts/model_weight/ --text_spotting_dir /your/textspotter/output/directory --output_dir_root /your/output/root/directory --symbol_info_json_file ./pipeline-scripts/automated_model_selection/symbol_info.json ```


This repository contains (1) pretrained models dir  (```--model_weights_dir ./pipeline-scripts/model_weight/``` )  and (2) a metadata used for selecting pretrained models per maps (```--symbol_info_json_file ./pipeline-scripts/automated_model_selection/symbol_info.json```) 


## Command Line Arguments:
--map_metadata_dir: Directory to map metadata that contains the 
information about coordinate of legend and map dimension. <br>
<!-- --cropped_legend_dir: Directory to save the cropped legend  <br>
--template_dir: Directory to the legend template  <br>
--processed_legend_dir: Directory to save the processed cropped legend  <br> -->
--map_patches_dir: Directory for input map patches directory per single map  <br>
--model_weights_dir: Directory to pretrained point feature detection model 
weights  <br>
--symbol_info_json_file : Json file used for selecting pretrained models based on text description on a map legend <br>
--output_dir_root: Root directory for output directory  <br>
--text_spotting_dir (Optional) : Directory of mapKurator outputs. This is optional, which is used for postprocessing point symbol module outputs <br>

 <br>
Feel free to replace "/your/map/directory" and other placeholders with the 
actual paths relevant to your project
 

