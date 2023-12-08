from ultralytics import YOLO
import os
import cv2
import json
from PIL import Image


config_root_path='CONFIG/DIR/PATH'
config_root_list=os.listdir(config_root_path)

"""Trainng a model""" 

for each_cfg in config_root_list:
      m_ep=30
      m_name=each_cfg.split('.')[0]
      cfg_path=os.path.join(config_root_path,each_cfg)
      model = YOLO('yolov8n.pt')
      results = model.train(
         data=cfg_path,
         imgsz=1024,
         epochs=m_ep,
         batch=32,
         conf=0.3,
         name=m_name)   


""" Evaluating a model with validation data """ 

model_result_root='MODEL/OUTPUT/DIR'
model_result_list=os.listdir(model_result_root)

for each_cfg in config_root_list:
      m_ep=30
      m_name=each_cfg.split('.')[0]
      model_path=os.path.join(model_result_root,m_name,'weights/best.pt')
      model = YOLO(model_path)
      results = model.val(conf=0.3)

""" Inferencing a model """ 

infer_data_input='TARGET/DATA/PATH'
infer_data_output='OUTPUT/DATA/PATH'

for each_cfg in config_root_list: 
      m_ep=30
      m_name=each_cfg.split('.')[0]
      model_path=os.path.join(model_result_root,m_name,'weights/best.pt')
      model = YOLO(model_path)
      val_data_path=infer_data_root
      output_path=infer_data_output
      if not os.path.isdir(output_path):
         os.mkdir(output_path)

  #saving a prediction result with BB (image)
      for val_img in os.listdir(val_data_path):
         if val_img.endswith('.jpg'):
            img_path=os.path.join(val_data_path,val_img)       
            results = model(img_path,conf=0.3)  
            for r in results:
               im_array = r.plot()  # plot a BGR numpy array of predictions
               im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
               im.show()  # show image
               out_img=os.path.join(output_path,val_img)
               im.save(out_img)  # save image

