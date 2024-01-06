# LDTR: Linear Object Detection Transformer for Accurate Graph Generation by Learning the N-hop Connectivity Information

### LDTR is a Transformer-based model to autormaticallt detect linear objects from the topographic or geological maps

## Docker image to train/testing LDTR
**Here is the command to run the LDTR docker image**
- Pull ldtr docker image from docker-hub
```
nvidia-docker pull weiweiduan/ldtr_pytorch:version0
```
- Run ldtr docker image
```
nvidia-docker run -t -i -v {local_dir}:{docker_dir} -p 8888:8888 weiweiduan/ldtr_pytorch:version0
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

## Testing Data Generation
```
 python generate_test_data.py --map_dir 'path/to/test_maps' --output_dir 'path/to/cropped_test_images'
```

## Use LDTR to detect desired linear objects on a single map
```
python test_darpa_map_conflation_shapely_mask_output_schema.py --cuda_visible_device 0 --config ./configs/usgs_railroads.yaml --checkpoint /path/to/trained_model/checkpoint_epoch.pt 
```

Please update './configs/usgs_railroads.yaml' for the path to the testing images 

## Use LDTR to detect desired linear objects on maps from a folder
```
python test_maps_from_folder.py --cuda_visible_device 0 --test_dir /path/to/map/folder --config ./configs/usgs_railroads.yaml --checkpoint /path/to/trained_model/checkpoint_epoch.pt 
```
