# Pytorch-YOLOv4

![](https://img.shields.io/static/v1?label=python&message=3.6|3.7&color=blue)
![](https://img.shields.io/static/v1?label=pytorch&message=1.4&color=<COLOR>)
[![](https://img.shields.io/static/v1?label=license&message=Apache2&color=green)](./License.txt)

## 0 目录树
```
├── README.md
├── dataset.py            dataset
├── demo.py               demo to run pytorch --> tool/darknet2pytorch
├── demo_darknet2onnx.py  tool to convert into onnx --> tool/darknet2pytorch
├── demo_pytorch2onnx.py  tool to convert into onnx
├── models.py             model for pytorch
├── train.py              train models.py
├── cfg.py                cfg.py for train
├── cfg                   cfg --> darknet2pytorch
├── data            
├── weight                --> darknet2pytorch
├── tool
│   ├── camera.py           a demo camera
│   ├── coco_annotation.py       coco dataset generator
│   ├── config.py
│   ├── darknet2pytorch.py
│   ├── region_loss.py
│   ├── utils.py
│   └── yolo_layer.py
```

## 1 准备
此模型使用**densenet**+**圆形先验框**实现樱桃检测，与苹果和橘子检测相比，仅仅有部分参数不同。使用方法与方形先验框代码相同。
### 1.1 权重
创建`checkpoints`文件夹，将三个模型权重解压至`checkpoints\apple_orange`、`checkpoints\cherry`和`checkpoints\squared_cherry`。其中`checkpoints\apple_orange`和`checkpoints\cherry`均为圆形先验框，`checkpoints\squared_cherry`为方形先验框。
### 1.2 环境
- pytorch 1.4+ 
- anaconda 3.7+
### 1.3 数据集
data.zip直接解压到根目录即可


## 2 使用
在根目录使用以下命令，或是直接使用vscode配置好的调试参数。
### 2.1 生成数据
```
PYTHONPATH=. python tool/format_convert.py -i=data/labelme/cherry -o=data/general/train.txt -t=yolo
```
### 2.2 训练
```
python train.py -g=0 -l=0.005 -dir=data/labelme/cherry -train_label_path=data/general/train.txt -classes=3
```

### 2.3 推断
```
python demo.py -weightfile=checkpoints/cherry/Yolov4_epoch500.pth -imgdir=data/labelme/cherry
```
### 3.1 Image input size for inference

Image input size is NOT restricted in `320 * 320`, `416 * 416`, `512 * 512` and `608 * 608`.
You can adjust your input sizes for a different input ratio, for example: `320 * 608`.
Larger input size could help detect smaller targets, but may be slower and GPU memory exhausting.

```py
height = 320 + 96 * n, n in {0, 1, 2, 3, ...}
width  = 320 + 96 * m, m in {0, 1, 2, 3, ...}
```

