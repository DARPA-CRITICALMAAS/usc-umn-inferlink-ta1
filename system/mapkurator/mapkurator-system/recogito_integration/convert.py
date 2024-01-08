import argparse
import geojson
import glob
import json
import numpy as np
import os

def transform_data(args):

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    # read mapkurator-system stitch module output.
    file_list = glob.glob(input_dir + '/*.geojson')
    file_list = sorted(file_list)
    if len(file_list) == 0:
        logging.warning('No files found for %s' % input_dir)
    
    # # read recogito input annotation format as a string. 
    # with open('/home/mapkurator-system/recogito_integration/annotation_format.json','r') as annotation_string:
    #     annotation = annotation_string.read()

    # Generate web annotations: https://www.w3.org/TR/annotation-model/
    for input_file in file_list:
        annotations = []
        with open(input_file) as img_geojson:
            img_data = geojson.load(img_geojson)
            for img_feature in img_data['features']:  
                polygon = img_feature['geometry']['coordinates'][0]
                svg_polygon_coords = ' '.join([f"{x},{y}" for x, y in polygon])
                annotation = {                
                    "@context": "http://www.w3.org/ns/anno.jsonld",
                    "id": "",
                    "type": "Annotation",
                    "body": [{
                        "type": "TextualBody",
                        "purpose": "transcribing",
                        "value": img_feature['properties']['text']
                        },
                        {
                        "type": "Dataset",
                        "format": "application/json",
                        "purpose": "documenting",
                        "value": {}
                        }],    
                    "target": {
                        "selector": {
                        "type": "SvgSelector",
                        "value": f"<svg><polygon points='{svg_polygon_coords}'></polygon></svg>"
                        }
                    }
                }
                annotations.append(annotation)
        
    # for input_file in file_list:
    #     annotations_list =[]
    #     with open(input_file) as img_geojson:
    #         img_data = geojson.load(img_geojson)
    #         for img_feature in img_data['features']:    
    #             obj = geojson.loads(annotation)
    #             #TODO: transform coordinate system if required.  
    #             #Copy over the data from mapkurator-system to recogito annotation format.
    #             obj['target']['selector']['value']="<svg><polygon points='"+str(np.array(img_feature['geometry']['coordinates']))+"' /></svg>"
    #             obj['body'][0]['value']= img_feature['properties']['text']
    #             annotations_list.append(obj)
    
        output_file = output_dir+input_file.split(input_dir)[1].split('.')[0]+'.json'
         
        with open(output_file, "w") as data_file:
            json.dump(annotations, data_file, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/home/mapkurator-results/modules_integration_test_02_17/stitch/mapkurator_integration_test/')
    parser.add_argument('--output_dir', type=str, default='/home/mapkurator-results/annotation_unit_test_01_03/')
    
    args = parser.parse_args()
    transform_data(args)
if __name__ == '__main__':

    main()