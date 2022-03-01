import os
import numpy as np

'''
model_path : 训练好的模型参数路径
anchors_path : 事先K-means找出的先验框尺寸参数路径
classes_path : 分类参数路径
model_image_size : 输入模型图像尺寸
confidence : 置信度阈值
iou : 非最大抑制IOU重合阈值
cuda : 是否使用GPU
'''
class YOLOConfig():
    _defaults = {
        "model_path": 'exmodel/resource/yolo4_voc_weights.pth',
        "anchors_path": 'exmodel/resource/yolo_anchors.txt',
        "classes_path": 'exmodel/resource/voc_classes.txt',
        "model_image_size": (416, 416, 3),
        "stride": [32, 16, 8],
        "confidence": 0.5,
        "batch_size" : 2,
        "iou": 0.3,
        "cuda": True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    '''初试化YOLO'''

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()

    '''加载分类参数'''

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    '''加载先验框参数'''

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape([-1, 3, 2])[::-1, :, :]