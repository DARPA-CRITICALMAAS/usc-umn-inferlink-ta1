def img2geo_pt(x, y, img_bbox, geo_bbox):
    """
    Convert a point from image coordinates to geographic coordinates.

    :param x: x-coordinate in the image
    :param y: y-coordinate in the image
    :param img_bbox: Bounding box of the image [left, right, top, bottom] in image coordinates
    :param geo_bbox: Bounding box of the geographic area [left, right, top, bottom] in geographic coordinates
    :return: (longitude, latitude)
     (-123.2, 49.3, -122.5, 48.9)  
    """
#     print(img_bbox)
    img_bbox = (img_bbox[0], img_bbox[2], img_bbox[1], img_bbox[3])
    img_x_min, img_y_min, img_x_max, img_y_max = img_bbox
    
#     print(geo_bbox)
    geo_bbox = (geo_bbox[0], geo_bbox[2], geo_bbox[1], geo_bbox[3])
    geo_lon_min, geo_lat_max, geo_lon_max, geo_lat_min = geo_bbox
#     print('--------------------')
#     print(img_bbox)
#     print(geo_bbox)
#     print((img_x_min, img_y_min), (geo_lon_min, geo_lat_min))
#     print((img_x_max, img_y_min), (geo_lon_max, geo_lat_min))
#     print((img_x_max, img_y_max), (geo_lon_max, geo_lat_max))
#     print((img_x_min, img_y_max), (geo_lon_min, geo_lat_max))
#     print('===========================')

    # Normalize the image coordinates to a [0, 1] scale
    x_normalized = (x - img_x_min) / (img_x_max - img_x_min)
    y_normalized = (y - img_y_min) / (img_y_max - img_y_min)

    # Interpolate to geographic coordinates
    lon = geo_lon_min + x_normalized * (geo_lon_max - geo_lon_min)
    lat = geo_lat_max - y_normalized * (geo_lat_max - geo_lat_min)

    return [lon, lat]


def img2geo_geometry(geometry, img_bbox, geo_bbox, geom_type):
    """
    Convert polygon/line/point from from image coordinates to geographic coordinates
    by calling img2geo_pt func, the input point is (col, row)
    :param geom_type: ['polyon', 'line', 'point']
    """
    if geom_type == 'polygon':
        geo_poly = [img2geo_pt(pt[0], pt[1], img_bbox, geo_bbox) for pt in geometry[0]]
        return [[geo_poly]]
    
    if geom_type == 'line':
        geo_line = [img2geo_pt(pt[1], pt[0], img_bbox, geo_bbox) for pt in geometry]
        return [geo_line]
    
    if geom_type == 'point':
        geo_point = img2geo_pt(geometry[0], geometry[1], img_bbox, geo_bbox)
        return geo_point
    
def img2qgis_geometry(geometry, geom_type):
    """
    Convert a point from (row, col) to (col, -row) for qgis visualization. 
    :param geom_type: ['polyon', 'line', 'point']
    """
    if geom_type == 'polygon':
        geo_poly = [[pt[0], -pt[1]] for pt in geometry[0]]
        return [[geo_poly]]
    
    if geom_type == 'line':
        geo_line = [[pt[1], -pt[0]] for pt in geometry]
        return [geo_line]
    
    if geom_type == 'point':
        geo_point = [geometry[0], -geometry[1]]
        return geo_point
        
    