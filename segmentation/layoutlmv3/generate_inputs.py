import os
import json
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

class GenerateInputs4LayoutLMv3:
    def __init__(self, image_input_path, legend_input_path, output_dir):
       self.image_input_path = image_input_path
       self.legend_input_path = legend_input_path
       self.output_dir = output_dir
       self.map_name = os.path.basename(image_input_path).split('.')[0]
       self.image_counter = 0
       self.annotation_counter = 0
    
    def crop_image(self):
        with open(self.legend_input_path, 'r') as f:
            legend_data = json.load(f)
        ptln_bboxes = []
        for item in legend_data['segments']:
            if item['class_label'] == 'legend_points_lines':
                ptln_bboxes.append(item['bbox'])
        cropped_images, cropped_image_names = [], []
        if ptln_bboxes != []:
            #load the image
            image_pil = Image.open(self.image_input_path)
            image = np.array(image_pil)
            #crop the image according to ptln_bbox
            for x, y, w, h in ptln_bboxes:
                x, y, w, h = int(x), int(y), int(w), int(h)
                cropped_images.append(image[y:y+h, x:x+w])
                cropped_image_names.append(f"{self.map_name}_{x}_{y}_{w}_{h}.png")
        return cropped_images, cropped_image_names, ptln_bboxes


    def generate_inputs4layoutlmv3(self):
        input_images, input_image_names, ptln_bboxes = self.crop_image()
        coco_images_dict = []
        coco_annotations_dict = []
        coco_categories_dict = []
        #create output image dir
        output_image_dir = os.path.join(self.output_dir, self.map_name)
        if not os.path.exists(output_image_dir):
            os.makedirs(output_image_dir)
        for ind, cropped_image in enumerate(input_images):
            coco_images_dict.append({"id": ind+1, "width": ptln_bboxes[ind][2], \
                                     "height": ptln_bboxes[ind][3], "file_name": input_image_names[ind]}) 
            #save cropped_image
            cropped_image_pil = Image.fromarray(cropped_image)
            output_image_path = os.path.join(output_image_dir, input_image_names[ind])
            cropped_image_pil.save(output_image_path)

        # generate the coco style json file
        coco_categories_dict = [{'id': 1, 'name': 'Symbol', 'supercategory': 'Figure'}, \
                            {'id': 2, 'name': 'Text', 'supercategory': 'Text'}, 
                            {'id': 3, 'name': 'Legend', 'supercategory': 'Legend'}]
        coco_json = {"images": coco_images_dict, "annotations": coco_annotations_dict,\
            "categories": coco_categories_dict}
        print(coco_json)
        
        #save the coco style json file
        output_json_path = os.path.join(self.output_dir, f'{self.map_name}.json')
        with open(output_json_path, "w") as outfile: 
            json.dump(coco_json, outfile)
        return output_image_dir
