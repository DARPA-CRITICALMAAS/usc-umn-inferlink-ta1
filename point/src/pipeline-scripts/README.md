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

``` 
python run_point_pipe.py --map_metadata_dir /your/metadata/directory --map_patches_dir /your/patches/directory/per/map --model_weights_dir model_weight/ --output_dir_root /your/output/root/directory --symbol_info_json_file utomated_model_selection_img_txt/data/symbol_info.json --gpu_id 0 --strike_model_dir strike_model_weights/ --cropped_legend_patches_dir /cropped/patches/outside-of-map-region/entire-maps-directory --image_based_model_weight automated_model_selection_img_txt/data/model.pt
```


This repository contains (1) pretrained models dir  (```--model_weights_dir model_weight/``` ),(2) a metadata used for selecting pretrained models per maps (```--symbol_info_json_file automated_model_selection_img_txt/data/symbol_info.json```) and (3) pretrained model file path for image-based automated model selection module (```--image_based_model_weight automated_model_selection_img_txt/data/model.pt``` ) 

Regarding the (```--dip_direct_model_path``` ), [here](https://drive.google.com/file/d/1C6TS_bb8KsxPtwA6KiXrtqFHigh7BFsq/view?usp=drive_link) is a URL to download the dip direction classification model weight file. 

## Command Line Arguments:

```
--map_dir : Directory containing entire map sheets
--map_metadata_dir: Directory of map metadata that contains the information about coordinate of legend and map dimension. 
--map_patches_dir: Directory of input map patches directory per single map 
--model_weights_dir: Directory of pretrained point feature detection model weights  
--symbol_info_json_file : Json file used for selecting pretrained models based on text description on a map legend 
--output_dir_root: Root directory for output directory
--gpu_id : Specifying GPU id for running the module
<!-- --strike_model_dir : Directory of pretrained strike model weight -->
--cropped_legend_patches_dir : Directory of cropped patches from outside of map region including entire maps
--image_based_model_weight : model weight path for image-based automated model selection module
--dip_direct_model_path : model weight file path for dip direction classification model 
--text_spotting_dir (Optional) : Directory of mapKurator outputs. This is optional, which is used for postprocessing point symbol module outputs

```

<!-- --cropped_legend_dir: Directory to save the cropped legend <br>
--template_dir: Directory to the legend template  <br>
--processed_legend_dir: Directory to save the processed cropped legend  <br> -->

Feel free to replace "/your/map/directory" and other placeholders with the 
actual paths relevant to your project
 
