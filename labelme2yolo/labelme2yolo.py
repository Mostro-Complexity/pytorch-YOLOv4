import os
import json
import PIL.Image
import PIL.ImageDraw
import numpy as np
from labelme2coco.utils import create_dir, list_jsons_recursively
from labelme2yolo.annotation_utils import shape_conversion


class labelme2yolo(object):
    def __init__(self, labelme_folder='', save_path='./new.json'):
        self.save_path = save_path

        # create save dir
        save_dir = os.path.dirname(save_path)
        create_dir(save_dir)

        # get json list
        _, self.labelme_json = list_jsons_recursively(labelme_folder)

        self.save()

    def sort_classes(self):
        self.classes = []
        for json_path in self.labelme_json:
            data = json.load(open(json_path, 'r', encoding='utf-8'))
            for shapes in data['shapes']:
                if shapes['label'] not in self.classes:
                    self.classes.append(shapes['label'])

        self.classes = sorted(self.classes)
        self.class_id = {c: i for i, c in enumerate(self.classes)}
        return self.classes, self.class_id

    def data_transfer(self):
        anno_patches = []
        for num, json_path in enumerate(self.labelme_json):
            data = json.load(open(json_path, 'r', encoding='utf-8'))
            image_path = data["imagePath"]

            image_patches = []
            for shapes in data['shapes']:
                label = self.class_id[shapes['label']]
                points = shape_conversion(shapes['points'], shapes['shape_type'])
                image_patches.append(self.object_annotation(*points, label))

            anno_patches.append(self.image_annotation(image_path, *image_patches))

        self.finalized = self.annotations(anno_patches)
        return self.finalized

    def object_annotation(self, *points):
        return ','.join([str(p) for p in points])

    def image_annotation(self, *patches):
        return ' '.join(patches)

    def annotations(self, patches):
        return '\n'.join(patches)

    def save(self):
        self.sort_classes()
        self.data_transfer()
        with open(self.save_path, 'w', encoding='utf-8') as fp:
            fp.write(self.finalized)


# type check when save json files
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


if __name__ == "__main__":
    labelme_folder = "tests/data/labelme_annot"
    save_json_path = "tests/data/test_coco.json"
    labelme2coco(labelme_folder, save_json_path)
