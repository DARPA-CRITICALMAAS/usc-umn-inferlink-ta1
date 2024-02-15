# Map Cropping Module
## Overview
Given the large dimensions of maps, this module crops a maps into patches for the follwoing modules (text spotting module and feature extraction module).

## Environment
- python 3.8
- OpenCV
- Numpy

## Running Command
```sh
python map2patch.py --input_dir /dir/to/maps --legend_dir /dir/to/legend --map_name 'AK_Dillingham' --patch_sizes 256 1000 1024 --strides 500 500 500 --output_dir /output/dir/to/cropped/patches
```
Notes: 
- The length of "patch_sizes" and "strides" must be the same. "strides" define cropping a patch when a sliding window moves x-pixel horizontally and vertically.
- In the "output\_dir", each patch size has a folder. The name convention of the folder is {map_name}\_g{patch\_size}\_s{stride}. In each folder, the name convention of each patch is {map_name}\_{row\}_{col}.png

