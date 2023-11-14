# LDTR: Linear Object Detection Transformer for Accurate Graph Generation by Learning the N-hop Connectivity Information

## LDTR's goal is to detect linear objects from the topographic or geological maps

## Docker imagery to train/testing LDTR
**Here is the command to run the docker imagery**

***Pull ldtr docker image from docker-hub
<code>docker nvidia-docker pull weiweiduan/ldtr_pytorch:versions

*** Run ldtr docker image
<code>sudo nvidia-docker run -t -i -v {local_dir}:{docker_dir} -p 8888:8888 ldtr_pytorch:version0</code>

## Testing Data Generation

<code> python generate_test_data.py </code>

## Use LDTR to detect desired linear objects on a single map

<code> python test_darpa_map_conflation_shapely_mask_output_schema.py --cuda_visible_device 0 --config ./configs/usgs_railroads.yaml --checkpoint /path/to/trained_model/checkpoint_epoch.pt </code>

Please update './configs/usgs_railroads.yaml' for the path to the testing images 

## Use LDTR to detect desired linear objects on maps from a folder

<code> python test_maps_from_folder.py --cuda_visible_device 0 --test_dir /path/to/map/folder --config ./configs/usgs_railroads.yaml --checkpoint /path/to/trained_model/checkpoint_epoch.pt </code>
