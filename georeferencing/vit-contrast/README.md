# Training Vision Transformer for Geologic/Topographic Map Matching


## Environment

This module was tested on an a100-8.

### Create from Conda Config

```
conda env create -f environment.yml
conda activate hugging
```

## Usage

### Train a Vision Transformer with Contrastive Learning

Descriptions of the inputs are as follows.

```
--topo_only: Turn on if training topo to topo
--topo_train_input_path: topo train dataset path
--topo_full_input_path: full topomap dataset
--test_input_path: topo test input path
--positives_dict_path: path to dictionary of anchors + positives
--model_save_path: model save path
--results_save_path: save path for correct + incorrect pairs
--topo_distances_mtx_path: save path for embedding matrix

--geo_distances_mtx_path: save path for embedding matrix
--geo_train_input_path: geologic map train dataset path
--geo_test_input_path: geologic map test dataset path

--model_type: resnet, c_vit-mae, vit-mae
--batch_size: batch size
--learning_rate: learning rate
--start_epoch: start epoch
--epochs: end epoch
--loss: triplet_margin or InfoNCE

--top_k: top-k for evaluation
```

