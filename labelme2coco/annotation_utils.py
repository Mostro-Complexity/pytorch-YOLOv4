import numpy as np


def circle2segmentation(points):
    angles = np.linspace(0, 2*np.pi, 50)
    radius = np.linalg.norm(points[1]-points[0])
    out_points = np.array([
        points[0, 0]+np.cos(angles)*radius,
        points[0, 1]+np.sin(angles)*radius
    ], dtype=int).transpose()
    return out_points


shape_methods = {
    'circle': circle2segmentation
}


def shape_conversion(points, shape_type):
    if isinstance(points, np.ndarray):
        pass
    elif isinstance(points, list):
        points = np.asarray(points)
    return shape_methods[shape_type](points).tolist()
