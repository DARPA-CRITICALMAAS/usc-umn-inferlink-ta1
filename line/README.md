# LDTR: Linear Object Detection Transformer for Accurate Graph Generation by Learning the N-hop Connectivity Information

### LDTR is a Transformer-based model to autormaticallt detect linear objects from the topographic or geological maps

## Docker image to run LDTR to extract lines
**Here is the command to run the LDTR docker image**
- Pull LDTR docker image from docker-hub
```
nvidia-docker pull weiweiduan/ldtr_pytorch:version0
```
- Run LDTR docker image
```
nvidia-docker run -t -i -v {local_dir}:{docker_dir} -p 8888:8888 weiweiduan/ldtr_pytorch:version0 /bin/bash
```
## Dokcerfile Usage
Dockerfile in this subdirectory is used to build the LDTR docker image. Please put requirements.txt and Dockerfile in the same folder when building the image. 
The Dockerfile includes all libraries and dependencies to run LDTR except the MultiScaleDeformableAttention package. Please first build the image and run the image. After that, please install MultiScaleDeformableAttention in the dokcer container following the steps below:
```
git clone https://github.com/fundamentalvision/Deformable-DETR.git
cd /root/Deformable-DETR/models/ops
sh ./make.sh
```
To verify the successful installation of MultiScaleDeformableAttention, you can execute the 'test.py' script located in the same directory. Running this script will return 'True' if MultiScaleDeformableAttention has been installed correctly.

## Use LDTR to detect desired linear objects on a single map
```
python run_line_extraction.py --config line/configs/usgs_railroads.yaml --map_legend_json /data/weiweidu/criticalmaas_data/github_test/gpt_outputs/AK_Dillingham.json --cuda_visible_device 3 --predict_vector True --cropped_image_dir /data/weiweidu/LDTR_criticalmaas/data/darpa/fault_lines/AK_Dillingham_g256_s100/raw/ --prediction_dir './pred_maps' --map_name 'AK_Dillingham.tif' --checkpoint /data/weiweidu/LDTR_criticalmaas_online_pos_neg/trained_weights/runs/fault_line/models/checkpoint_epoch=180.pt 
```
```
--config config file is in this subdirectory, line/configs/usgs_railroads.yaml
--map_legend_json the output from "segmentation/legend_item_description_segment" module
--predict_vector set True the output is geojson
--predict_raster set True the output is png
--cropped_image_dir the output from "system/image_crop" module
--prediction_dir the directory to save the extraction results
--map_name the map name, please add ".tif" in the end
--checkpoint /path/to/trained_model 
```
The pretrained model can be downloaded from [link](https://drive.google.com/drive/folders/1EA3PyR2d1m9S-xQvtr3YRmy9UvBiJVW3?usp=sharing)
