# Point Feature Pipleline

## Instruction:

To run the project, use the following command:

``` python run_final.py --map_dir /your/map/directory --map_metadata_dir /your/metadata/directory --cropped_legend_dir /your/legend/directory --template_dir /your/template/directory --processed_legend_dir /your/processed/legend/directory --map_patches_dir /your/patches/directory --model_weights_dir /your/weights/directory --output_dir_root /your/output/root/directory ```

## Command Line Arguments:
--map_dir: Directory to the original map data ended with .tif <br>
--map_metadata_dir: Directory to map metadata that contains the 
information about coordinate of legend and map dimension. <br>
--cropped_legend_dir: Directory to save the cropped legend  <br>
--template_dir: Directory to the legend template  <br>
--processed_legend_dir: Directory to save the processed cropped legend  <br>
--map_patches_dir: Root directory for input map patches directory  <br>
--model_weights_dir: Directory to pretrained point feature detection model 
weights  <br>
--output_dir_root: Root directory for output directory  <br>
 <br>
Feel free to replace "/your/map/directory" and other placeholders with the 
actual paths relevant to your project.
 

