# Map Area Segmentation for Geological Maps

## Environment


### Create from Separate Steps

```
pip install -r requirement.txt
```


## Usage

```
python map_area_segmentation.py --input_path xxx --binary_output_path xxx --json_output_path xxx --intermediate_dir Intermediate --version 0 --log xxx.txt
```

```
--input_path: (str) path to the source map tif.
--binary_output_path: (str) path to the binary output of map area tif.
--json_output_path: (str) path to the json output of map area.
--intermediate_dir: (str) dir to store binary output and other intermediate images.
--version: (str, optional) a value indicating the version. Default to '0'.
--log: (str, optional) path to the logging txt. Default to 'log_file.txt'.
```
