# -*- coding: utf-8 -*-
'''
@Time          : 2020/05/06 21:05
@Author        : Tianxiaomo
@File          : Cfg.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''
import os
import torch
from easydict import EasyDict


def default_config():
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    Cfg = EasyDict()

    Cfg.use_darknet_cfg = False
    Cfg.cfgfile = os.path.join(_BASE_DIR, 'cfg', 'yolov4.cfg')

    Cfg.batch = 1
    Cfg.subdivisions = 1
    Cfg.width = 1280
    Cfg.height = 800
    Cfg.channels = 3
    Cfg.momentum = 0.949
    Cfg.decay = 0.0005
    Cfg.angle = 0
    Cfg.saturation = 1.5
    Cfg.exposure = 1.5
    Cfg.hue = .1

    Cfg.learning_rate = 0.00261
    Cfg.burn_in = 1000
    Cfg.max_batches = 500500
    Cfg.steps = [400000, 450000]
    Cfg.anchors = torch.tensor([
        [12, 16],
        [19, 36],
        [40, 28],
        [36, 75],
        [76, 55],
        [72, 146],
        [142, 110],
        [192, 243],
        [459, 401]
    ], dtype=torch.float32)
    Cfg.policy = Cfg.steps
    Cfg.scales = .1, .1
    Cfg.workers = 0

    Cfg.cutmix = 0
    Cfg.mosaic = 0

    Cfg.letter_box = 0
    Cfg.jitter = 0.15
    Cfg.classes = 3
    Cfg.track = 0
    Cfg.w = Cfg.width
    Cfg.h = Cfg.height
    Cfg.flip = 1
    Cfg.blur = 0
    Cfg.gaussian = 0
    Cfg.boxes = 1200  # box num
    Cfg.TRAIN_EPOCHS = 300
    Cfg.train_label = os.path.join(_BASE_DIR, 'data', 'general', 'train.txt')
    Cfg.val_label = os.path.join(_BASE_DIR, 'data', 'general', 'val.txt')
    Cfg.TRAIN_OPTIMIZER = 'adam'
    Cfg.VALIDATE_INTERVAL = 20
    '''
    image_path1 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
    image_path2 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
    ...
    '''

    if Cfg.mosaic and Cfg.cutmix:
        Cfg.mixup = 4
    elif Cfg.cutmix:
        Cfg.mixup = 2
    elif Cfg.mosaic:
        Cfg.mixup = 3
    else:
        Cfg.mixup = 0

    Cfg.checkpoints = os.path.join(_BASE_DIR, 'checkpoints')
    Cfg.TRAIN_TENSORBOARD_DIR = os.path.join(_BASE_DIR, 'log')

    Cfg.iou_type = 'iou'  # 'giou', 'diou', 'ciou'

    Cfg.keep_checkpoint_max = 10

    Cfg.pretrained = None

    return Cfg
