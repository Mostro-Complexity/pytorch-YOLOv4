import numpy as np


def circle2coordinates(points):
    radius = np.linalg.norm(points[1]-points[0])
    coord = points[0].round().astype(np.int64).tolist()
    coord.append(int(round(radius)))
    return coord

def rectangle2coordinates(points):
    return points.reshape(-1)

shape_methods = {
    'circle': circle2coordinates,
    'rectangle':rectangle2coordinates
}


def shape_conversion(points, shape_type):
    if isinstance(points, np.ndarray):
        pass
    elif isinstance(points, list):
        points = np.asarray(points)
    return shape_methods[shape_type](points)
