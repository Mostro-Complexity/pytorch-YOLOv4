import numpy as np


def circle2segmentation(points):
    angles = np.linspace(0, 2*np.pi, 50)
    radius = np.linalg.norm(points[1]-points[0])
    out_points = np.array([
        points[0, 0]+np.cos(angles)*radius,
        points[0, 1]+np.sin(angles)*radius
    ], dtype=int).transpose()
    return out_points


def rectangle2segmentation(points):
    left_top = points[0]
    left_bottom = np.array([points[0, 0], points[1, 1]])
    right_bottom = points[1]
    right_top = np.array([points[1, 0], points[0, 1]])
    return np.stack([left_top, left_bottom, right_bottom, right_top])


shape_methods = {
    'circle': circle2segmentation,
    'rectangle': rectangle2segmentation
}


def shape_conversion(points, shape_type):
    if isinstance(points, np.ndarray):
        pass
    elif isinstance(points, list):
        points = np.asarray(points)
    return shape_methods[shape_type](points).tolist()
