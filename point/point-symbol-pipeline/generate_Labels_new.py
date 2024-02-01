import os
import json
from PIL import Image
import PIL.Image
import glob

PIL.Image.MAX_IMAGE_PIXELS = None

def crop_and_save_image(image_path, coordinates, output_path):
    # Open the image
    original_image = Image.open(image_path)

    # Extract coordinates
    x1, y1, x2, y2 = coordinates

    upper_left = (min(x1, x2), min(y1, y2))
    bottom_right = (max(x1, x2), max(y1, y2))

    # Crop the image
    cropped_image = original_image.crop((*upper_left, *bottom_right))

    # Save the cropped image as JPEG
    cropped_image.save(output_path, "JPEG")

def crop_legned(folder_path, metadata_path, output_path):

    for filename in os.listdir(folder_path):
        if filename.endswith(".tif"):
            map_ = filename.split(".")[0]
            map_metadata = map_ + "_point.json"
            map_path = os.path.join(folder_path, filename)
            map_metadata_path = os.path.join(metadata_path, map_metadata)
            with open(map_metadata_path, 'r') as file:
                data = json.load(file)
                for coord_list, pt_dict in data.items():
                    if coord_list not in ['map_content_box', 'map_dimension']:
                        modified_string = pt_dict['symbol name'].replace(" ", "_")
                        coordinates = json.loads(coord_list)
                        img_name = map_ + "_label_" + modified_string + ".jpeg"
                        output = os.path.join(output_path, img_name)
                        crop_and_save_image(map_path, coordinates, output)

