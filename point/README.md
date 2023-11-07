# Point symbol extraction based on object detection approach 

## Model 
For extracting point symbol classes whihc are seen during the training process, we leveraged YOLOv8 object detection model.
More information about the model, please refer to <a href='https://github.com/ultralytics/ultralytics'> this </a>

### model/run_yolo8.py 
This scirpt includes training, validating and prediction process by running following command 
```
python model/run_yolo8.py
```
Make sure to place the dataset and configuration file in the right path  

## Data preporcessing 
We pre-processed the data for formatting as an input of the YOLOv8 object detection model


