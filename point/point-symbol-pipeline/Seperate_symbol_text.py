from PIL import Image, ImageDraw
import pytesseract
import matplotlib.pyplot as plt
import sys
import easyocr
import cv2
import numpy as np
import os

def crop_image(img):
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_copy=img.copy()
    lower=np.array([10,10,10])
    higher=np.array([250,250,250])
    mask=cv2.inRange(img, lower, higher)
    contours, hierarchy = cv2.findContours(image= mask,
                                           mode=cv2.RETR_EXTERNAL,
                                           method=cv2.CHAIN_APPROX_NONE)
    sorted_contours=sorted(contours, key=cv2.contourArea, reverse= True)

    cont_img=cv2.drawContours(image=img, contours=sorted_contours, contourIdx=0,
                              color=(0,255,0),thickness=3)
    c=sorted_contours[0]
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(img=img, pt1=(x,y), pt2=(x+w,y+h), color=(0,255,0), thickness=3)
    cropped_image=img_copy[y:y+h+1, x:x+w+1]
    return cropped_image

def clean_background_and_remove_text(input_path, output_path, target_color, threshold=0, crop=False, use_gpu=False):

    image_files = [f for f in os.listdir(input_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    for image_name in image_files:
        image_path = os.path.join(input_path, image_name)

        # Read the image using OpenCV
        img = cv2.imread(image_path)

        # Define the target color
        target_color = np.array(target_color)

        # Find pixels close to the target color
        close_pixels = np.all(np.abs(img - target_color) < threshold, axis=-1)

        # Update the pixels to the target color
        img[close_pixels] = target_color

        reader = easyocr.Reader(['en'], gpu=use_gpu)
        result = reader.readtext(img)
        for detection in result:
            # print(detection)
            box = np.array(detection[0])
            points = box.astype(int)

            # Replace text region with background color (white in this case)
            cv2.rectangle(img, tuple(points[0]), tuple(points[2]), (255, 255, 255), thickness=cv2.FILLED)

        try:
            if crop:
                img = crop_image(img)

            save_path = os.path.join(output_path, image_name)
            cv2.imwrite(save_path, img)

        except:
            print(f"Error processing {image_path}")





