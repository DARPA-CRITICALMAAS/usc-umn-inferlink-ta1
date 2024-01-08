# Text Spotting Module
## This module aims to extract all texts from maps. The module is built on MapKurator, a text sptting tool developed from the Spatial Computing Lab from UMN.

## Environment 
Please refer to [MapKurator Doc](https://knowledge-computing.github.io/mapkurator-doc/#/docs/install1) to setup the enviroment to run the text spotting module. It provides the docker image, which is build on nvidia/cuda:11.3.0-devel-ubuntu18.04.

The pretrained model is located at /home/spotter\_v2/PALEJUN/weights/synthmap\_pretrain/model_final.pth in the docker image.

## Running Command
🔴 Please make sure that you set up the environment for MapKurator before running the command below \
🔴 Please use the MapKurator docker as the running environment \
🔴 Please use the source codes for text spotting in this folder
```
cd map_kurator_system
python run_text_spotting.py --map_kurator_system_dir /data/weiweidu/criticalmaas_data/github_test/mapkurator/mapkurator-system/ --input_dir_path /data/weiweidu/criticalmaas_data/github_test/criticalmaas_data/github_test/output_crop/AK_Dillingham_g1000_s500/ --expt_name mapKurator_test --module_text_spotting --text_spotting_model_dir /data/weiweidu/criticalmaas_data/github_test/mapkurator/spotter_v2/PALEJUN/ --spotter_model spotter_v2 --spotter_config /data/weiweidu/criticalmaas_data/github_test/mapkurator/spotter_v2/PALEJUN/configs/PALEJUN/SynthMap/SynthMap_Polygon.yaml --spotter_expt_name test --module_img_geojson --output_folder /data/weiweidu/criticalmaas_data/github_test/mapkurator/mapkurator-test-images/output/ --model_weight_path /data/weiweidu/mapkurator/spotter_v2/PALEJUN/weights/synthmap_pretrain/model_final.pth
```

```
Arguments:
--map_kurator_system_dir The directory to map_kurator_system for MapKurator
--input_dir_path The directory for the cropped images
--text_spotting_model_dir The directory to the spotting model for MapKurator
--spotter_config The path to the config file for MapKurator
--output_folder The directory to save the text spotting results in geojson
--model_weight_path The path to the pretrained model
```
