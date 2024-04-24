import numpy as np
import cv2
import json
import rasterio
import os
from rasterio.transform import from_origin
from rasterio.features import rasterize
from shapely.geometry import Polygon


def geojson_to_tif(info_id, info_subset):
    path_to_input_geojson = info_subset[0]
    path_to_output_tif = info_subset[1]
    example_input_shape = info_subset[2]
    #placeholder = np.zeros((example_input.shape[0], example_input.shape[1]), dtype=np.uint8)

    if os.path.isfile(path_to_input_geojson) == False:
        placeholder = np.ones((example_input_shape[0], example_input_shape[1]), dtype=np.uint8)*255
        cv2.imwrite(path_to_output_tif, placeholder)
        return False, path_to_output_tif


    # Load JSON data
    with open(path_to_input_geojson, 'r') as f:
        json_data = json.load(f)
        

    # Define the dimensions of the image
    width = example_input_shape[1]
    height = example_input_shape[0]

    # Create a transform for the raster image
    transform = from_origin(0, example_input_shape[0], 1, 1)  # Assuming each pixel represents 1 unit

    # Define the polygons
    polygons = [(Polygon(poly['geometry']['coordinates'][0]), 255) for feature in json_data for poly in feature['polygon_features']['features']]

    if len(polygons) > 0:
        # Create the raster image
        with rasterio.open(
            path_to_output_tif,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype='uint8',
            crs='+proj=latlong',
            transform=transform,
        ) as dst:
            dst.write(rasterize(polygons, out_shape=(height, width), transform=transform), 1)
        

        this_poly_output = cv2.imread(path_to_output_tif)
        this_poly_output = np.flipud(this_poly_output)
        cv2.imwrite(path_to_output_tif, this_poly_output)
    else:
        placeholder = np.zeros((example_input_shape[0], example_input_shape[1]), dtype=np.uint8)
        cv2.imwrite(path_to_output_tif, placeholder)
    

    return True, path_to_output_tif
