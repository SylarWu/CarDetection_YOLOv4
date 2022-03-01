import cv2
import numpy as np
import colorsys
import os
import torch
import torch.nn as nn
from exmodel.yolo4 import YoloBody
import torch.backends.cudnn as cudnn
from PIL import Image,ImageFont, ImageDraw
from torch.autograd import Variable
from utils.utils import non_max_suppression, bbox_iou, DecodeBox,letterbox_image,yolo_correct_boxes
import time
from config.config import YOLOConfig


class YOLO(object):

    '''初试化YOLO'''
    def __init__(self, **kwargs):
        self.config = YOLOConfig()
        self.generate()

    '''生成模型'''
    def generate(self):
        
        self.net = YoloBody(len(self.config.anchors[0]),len(self.config.class_names)).eval()

        # 加快模型训练的效率
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() and self.config.cuda else 'cpu')
        state_dict = torch.load(self.config.model_path, map_location=device)
        self.net.load_state_dict(state_dict)
        
        if self.config.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()
    
        print('Finished!')

        image_size = (self.config.model_image_size[1], self.config.model_image_size[0])

        self.yolo_decodes = []
        for i in range(3):
            self.yolo_decodes.append(DecodeBox(self.config.anchors[i],
                                               len(self.config.class_names),
                                               self.config.batch_size,
                                               image_size,
                                               (image_size[0] // self.config.stride[i], image_size[1] // self.config.stride[i]),
                                               self.config.cuda))


    '''生成检测'''
    def generate_detections(self, images):
        with torch.no_grad():
            outputs = self.net(images)
        output_list = []
        for i in range(3):
            output_list.append(self.yolo_decodes[i](outputs[i]))
        output = torch.cat(output_list, 1)
        batch_detections = non_max_suppression(output, len(self.config.class_names),
                                               conf_thres=self.config.confidence,
                                               nms_thres=self.config.iou)
        return batch_detections



if __name__ == '__main__':
    yolo = YOLO()