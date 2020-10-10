from __future__ import absolute_import


from labelme2yolo.labelme2yolo import labelme2yolo


def convert(labelme_folder: str, save_json_path: str):
    labelme2yolo(labelme_folder, save_json_path)
