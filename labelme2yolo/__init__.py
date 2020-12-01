from __future__ import absolute_import


from labelme2yolo.labelme2yolo import labelme2yolo


def convert(labelme_folder: str, save_json_path: str):
    return labelme2yolo(labelme_folder, save_json_path)


def save_classes(yolo, save_path):
    with open(save_path, 'w') as f:
        for c in yolo.classes:
            f.write('{:s}\n'.format(c))
