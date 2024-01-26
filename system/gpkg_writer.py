def image_to_geo(x, y, img_bbox, geo_bbox):
    """
    Convert image coordinates to geographic coordinates.

    :param x: x-coordinate in the image
    :param y: y-coordinate in the image
    :param img_bbox: Bounding box of the image (top-left, bottom-right) in image coordinates
    :param geo_bbox: Bounding box of the geographic area (top-left, bottom-right) in geographic coordinates
    :return: (longitude, latitude)
    """

    img_x_min, img_y_min, img_x_max, img_y_max = img_bbox
    geo_lon_min, geo_lat_max, geo_lon_max, geo_lat_min = geo_bbox

    # Normalize the image coordinates to a [0, 1] scale
    x_normalized = (x - img_x_min) / (img_x_max - img_x_min)
    y_normalized = (y - img_y_min) / (img_y_max - img_y_min)

    # Interpolate to geographic coordinates
    lon = geo_lon_min + x_normalized * (geo_lon_max - geo_lon_min)
    lat = geo_lat_max - y_normalized * (geo_lat_max - geo_lat_min)

    return lon, lat

# Example usage
x, y = 100, 50  # Image coordinates
img_bbox = (0, 0, 1024, 768)  # Image bounding box: top-left and bottom-right corners
geo_bbox = (-123.2, 49.3, -122.5, 48.9)  # Geographic bounding box: top-left and bottom-right corners

lon, lat = image_to_geo(x, y, img_bbox, geo_bbox)
print("Geographic Coordinates: Longitude = {}, Latitude = {}".format(lon, lat))
