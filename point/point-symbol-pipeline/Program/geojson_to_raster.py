import os
import json
import geopandas as gpd
from PIL import Image
import PIL.Image
import numpy as np
import glob
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
import cv2
import matplotlib.pyplot as plt


PIL.Image.MAX_IMAGE_PIXELS = None

def write_to_tif(out_file_path, output_tif):
    print('wrote to',out_file_path)

    cv2.imwrite(out_file_path, output_tif)

    # convert the image to a binary raster .tif
    raster = rasterio.open(out_file_path)
    transform = raster.transform
    array     = raster.read(1)
    crs       = raster.crs
    width     = raster.width
    height    = raster.height

    raster.close()
    with rasterio.open(out_file_path, 'w',
                    driver    = 'GTIFF',
                    transform = transform,
                    dtype     = rasterio.uint8, # pred_binary_raster.dtype, # rasterio.uint8,
                    count     = 1,
                    compress  = 'lzw',
                    crs       = crs,
                    width     = width,
                    height    = height) as dst:

        dst.write(array, indexes=1)
        dst.close()

    print('wrote output to ',out_file_path)

def geojson_to_raster(metadata_path, geojson_path, output_path):
    for filename in os.listdir(geojson_path):
        if filename.endswith(".geojson"):
            pixel_val = 0
            seen_pt_names = set()

            map_ = filename.split(".")[0]
            # print(map_)
            mapname = map_ + ".tif"
            map_metadata_name = map_ + "_point.json"
            file_path = os.path.join(metadata_path, map_metadata_name)
            print(file_path)
            with open(file_path, 'r') as file:
                data = json.load(file)

            width, height = data['map_dimension']
            print((width, height))
            raster_layer = np.zeros((height, width))

            geojson = os.path.join(geojson_path, filename)
            with open(geojson, 'r') as file:
                data = json.load(file)
                point_features_list = data['features']
                for point_feature_info in point_features_list:
                    pt_name = point_feature_info['properties']['type']
                    coord = point_feature_info['geometry']['coordinates']
                    x, y = int(coord[0]), int(coord[1])

                    if pt_name not in seen_pt_names:
                        pixel_val += 1

                    raster_layer[x, y] = pixel_val
                    seen_pt_names.add(pt_name)

            print(raster_layer)
            print(np.unique(raster_layer))
            tif_file_path = os.path.join(output_path, mapname)
            write_to_tif(tif_file_path, raster_layer)

metadata_path = "/Users/dong_dong_dong/Downloads/Darpa/data/sample/map_metadata"
stitch_output_dir = "/Users/dong_dong_dong/Downloads/Darpa/data/sample/model_output_pipeline/stitch"
raster_output_path = "/Users/dong_dong_dong/Downloads/Darpa/data/sample/model_output_pipeline/raster_layer"
geojson_to_raster(metadata_path, stitch_output_dir, raster_output_path)
