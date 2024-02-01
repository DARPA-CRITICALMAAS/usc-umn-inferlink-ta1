## Point symbol pipeline 


### Command line 

```python run_final.py --map_dir /your/map/directory --map_metadata_dir /your/metadata/directory --cropped_legend_dir /your/legend/directory --template_dir /your/template/directory --processed_legend_dir /your/processed/legend/directory --map_patches_dir /your/patches/directory --model_weights_dir /your/weights/directory --output_dir_root /your/output/root/directory```

--map_dir: Directory to the original map data ended with .tif

--map_metadata_dir: Directory to map metadata that contains the information about coordinate of legend and map dimension.

--cropped_legend_dir: Directory to save the cropped legend

--template_dir: Directory to the legend template

--processed_legend_dir: Directory to save the processed cropped legend

--map_patches_dir: Root directory for input map patches directory

--model_weights_dir: Directory to pretrained point feature detection model weights

--output_dir_root: Root directory for output directory


