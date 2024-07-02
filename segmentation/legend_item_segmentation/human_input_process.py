import os
import cv2
import json

class HumanInputProcess:
    def __init__(self, human_input: str):
        self.human_input = human_input

    def get_human_input(self):
        if os.path.exists(self.human_input):
            with open(self.human_input) as f:
                return json.load(f)
        return None
    
    def get_human_input_path(self):
        return self.human_input

    def process_human_input(self):
        human_input = self.get_human_input()
        if human_input is None:
            return None, None, None
        label = human_input['label']
        bbox = human_input['px_bbox']
        description = human_input['description']
        label = human_input['label']
        human_output_dict = {
            "label": label,
            "points": bbox,
            "group_id": "null",
            "shape": "rectangle",
        }
        geojson_human_output_dict = {
            "type": "Feature",
            "properties": {
                "name": human_input['abbreviation'],
                "abbreviation": human_input['abbreviation'],
                "id": human_input['legend_id'],
                "map_unit": "null",
                "color": human_input['color'],
                "pattern": human_input['pattern'],
                "description": description,
                "category": human_input['category']
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[bbox[0],bbox[1]], [bbox[2],bbox[1]],\
                                [bbox[2],bbox[3]], [bbox[0],bbox[3]], \
                                [bbox[0],bbox[1]]]
        }
        }
        category = human_input['category']
        return human_output_dict, geojson_human_output_dict, category

    def generate_module_output(self, output_dir: str):
        human_output, geojson_human_output, legend_type = self.process_human_input()
        output_dict = {
            "version": 1.0.1,
            "flags": "null",
            "shapes": []
        }
        if human_output is not None and legend_type != "polygon":
            output_dict[shapes].append(human_output)
        output_path = os.path.join(output_dir, f'{map_name}_PolygonType_internal.json')
        with open(output_path, 'w') as f:
            json.dump(output_dict, f)
        
        geojson_output_dict = {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {
                    "name": "urn:ogc:def:crs:EPSG::3857"
                }
            },
            "features": []
        }
        if geojson_human_output is not None and legend_type != "polygon":
            geojson_output_dict[shapes].append(geojson_human_output)
        geojson_output_path = os.path.join(output_dir, f'{map_name}_PolygonType.geojson')
        with open(geojson_output_path, 'w') as f:
            json.dump(geojson_output_dict, f)
        return output_path
