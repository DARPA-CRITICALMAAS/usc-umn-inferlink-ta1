# GeoPackge Generation Module
## Overview
This module combines outputs from other modules in th system into a single GeoPackage output. 

## Environment
- python 3.10
- geopandas
- pandas
- CriticalMAAS TA1 geopackage library (developed by the TA4 team). Please refer [Link](https://github.com/DARPA-CRITICALMAAS/ta1-geopackage/tree/47f585a0386dd5db3e7a9d96cc53d1e1b4f2ce10) for installation 


## Running Command
```sh
python run_gpkg_writer.py --output_dir /dir/to/save/gpkg --map_name '22253_25695' --layout_output_dir /dir/to/outputs/of/legend/item-description/sgement/module --georef_output_dir /dir/to/outputs/of/georef/module --poly_output_dir /dir/to/outputs/of/polygon/module --ln_output_dir /dir/to/outputs/of/line/module --pt_output_dir /dir/to/outputs/of/point/module --nongeoref_map_dir /dir/to/tif/map --georef_map_output /dir/to/save/georeferenced/map
```
Notes: 
- This module has three outputs: 1. georeferenced tif map, 2. GeoPackage in EPSG:4236 and 3.  GeoPackage in image coordinates
- The inputs for this module follow the file structure below:
```bash
outputs
|
|__tif_maps
|   |
|   |__map_name1.tif
|
|__legend_item_segment_outputs
|		|
|		|__map_name1_line.json
|
|___georeference_outputs
|		|
|		|___map_name1.json
|			
|
|___polygon_outputs
|		|
|		|____map_name1
|			|
|			|____poly_feature1.geojson 
|___point_outputs
|	|				
|	|____map_name1.geojson 
|
|___line_outputs
|	|
|	|____map_name1
|		|					
|		|____line_feature1.geojson
```
