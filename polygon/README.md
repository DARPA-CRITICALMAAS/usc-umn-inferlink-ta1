# LOAM

LOAM: Polygon Extraction from Raster Map


## Environment

### Create from Conda Config

```
conda env create -f environment.yml
conda activate loam
```

### Create from Separate Steps
```
conda create -n loam python=3.9.16
conda activate loam

conda install pytorch torchvision==1.13.1 torchaudio==0.13.1 cudatoolkit=11.7 -c pytorch
pip install -r requirements.txt
```


## Usage

### Extracting with a Pre-trained Model

We provide the link to our pre-trained model used for one-click procedure in `\LOAM\checkpoints\checkpoint_epoch2.pth_link.txt`, or access here to [Google Drive](https://drive.google.com/file/d/16N-2NbtqLSNCU83J5Iyw4Ca8A2f7aIqi/view?usp=sharing).

Setup `targeted_map.csv` for the index to the targeted raster maps, and run the following to exploit the pre-trained polygon-recognition model for extracting polygonal features from raster maps.

```
python loam_handler.py --data_dir --data_groundtruth_dir --solutiona_dir --targeted_map_list targeted_map.csv
```

or with some custom settings:

```
python loam_handler.py --data_dir xxx --solutiona_dir xxx --targeted_map_list targeted_map.csv
cd LOAM
python loam_handler.py --data_dir xxx --data_groundtruth_dir xxx --model_inference TRUE
```

```
--data_dir: (str) path to the source map tif.
--data_groundtruth_dir: (str) path to the groundtruth of map key tif. (this is to support evaluation)
--solutiona_dir: (str) path to the generated intermediate bitmaps.
--targeted_map_list: (str) a csv file that lists the map name.
--map_preprocessing: (bool) whether perform map cropping under metadata preprocessing. (default to True)
--generate_boundary_extraction: (bool) whether perform boundary extraction under metadata preprocessing. (default to True)
--printing_auxiliary_information: (bool) whether perform auxiliary-info extraction under metadata preprocessing. (default to True)
--preprocessing_recoloring: (bool) whether perform color-set matching under metadata preprocessing. (default to True)
--model_inference: (bool) to perform either metadata preprocessing (False) or polygon-recognition model (True). (default to False)
```
