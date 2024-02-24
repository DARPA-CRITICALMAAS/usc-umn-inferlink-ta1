# Map Cropping Module
## Overview
Given the large dimensions of maps, this module crops a maps into patches for the follwoing modules (text spotting module and feature extraction module).

## Environment
- python 3.8
- OpenCV
- Numpy

## Running Command
```sh
python map2patch.py --input_dir /dir/to/maps --legend_dir /dir/to/legend --map_name 'AK_Dillingham' --patch_sizes 256 1000 1024 --strides 500 500 500 --only_crop_map_area False True True --output_dir /output/dir/to/cropped/patches
```
Notes: 
- The length of "patch\_sizes", "strides" and "only\_crop\_map\_area" must be the same. 
- "strides" define cropping a patch when a sliding window moves x-pixel horizontally and vertically.
- "only\_crop\_map\_area" determines whethe the cropped patches include the map legend area. The text spotting module requires patches in both map content and legend area, whereas the feature extraction modules only require pathces from the map content area.
- In the "output\_dir", each patch size has a folder. Each folder follows the naming convention {map_name}_g{patch_size}_s{stride}_wo_legend for patches from the map content area only, and {map_name}_g{patch_size}s{stride}_w_legend for patches from both the map content and legend area. Within each folder, patches are named according to the convention {map_name}{row}{col}.png."

