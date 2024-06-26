import os
import cv2
import json

class HumanInputProcess:
    def __init__(self, human_input: str, legend_input: str, map_path: str):
        self.human_input = human_input
        self.legend_input = legend_input
        self.map_path = map_path

    def get_human_input(self):
        if os.path.exists(self.human_input):
            with open(self.human_input) as f:
                return json.load(f)
        return None
    def get_legend_input(self):
        if os.path.exists(self.legend_input):
            with open(self.legend_input) as f:
                return json.load(f)
        return None
    
    def get_human_input_path(self):
        return self.human_input

    def process_human_input(self):
        human_input = self.get_human_input()
        if human_input is None:
            return None
        bbox = human_input['px_bbox']
        description = human_input['description']
        label = human_input['label']
        human_output_dict = {
            str(bbox): {
            'description': description,
            'symbol name': label
            }
        }
        category = human_input['category']
        return human_output_dict, category

    def generate_module_output(self, output_dir: str):
        human_output, legend_type = self.process_human_input()
        
        legend_output = self.get_legend_input()
        for item in legend_output['segments']:
            if item['class_label'] == 'legend_polygons':
                poly_bbox = item['bbox']
            if item['class_label'] == 'legend_points_lines':
                ptln_bbox = item['bbox']
            if item['class_label'] == 'map':
                map_content_bbox = item['bbox'] 
        # load the map image
        map_image = cv2.imread(self.map_path)
        map_height, map_width, _ = map_image.shape

        output_dict = {
            "map_content_box": map_content_bbox,
            "poly_box": poly_bbox,
            "ptln_box": ptln_bbox,
            "map_dimension":[map_width, map_height]            
        }
        # combine human_output and output_dict
        output_dict.update(human_output)
        map_name = os.path.splitext(os.path.basename(self.map_path))[0] 
        output_path = os.path.join(output_dir, f'{map_name}_{legend_type}.json')
        with open(output_path, 'w') as f:
            json.dump(output_dict, f)
        return output_path
