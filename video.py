#-------------------------------------#
#       调用摄像头检测
#-------------------------------------#
from yolo import YOLO
from PIL import Image
import numpy as np
import cv2
import time
from parallel.model import *
import sys
import os
from pathlib import Path

def init_frame_processors(frame_producer, num_processor, max_inner_queue_size):
    # 初试化同步锁
    locks = [threading.Lock() for __ in range(num_processor)]
    for __ in range(num_processor - 1):
        locks[__].acquire()

    frame_processors = [
        FrameProcessor("FrameProcessor-" + str(__ + 1), locks[__ - 1], locks[__], max_inner_queue_size, frame_producer)
        for __ in range(num_processor)
    ]

    return frame_processors




if __name__=="__main__":
    if len(sys.argv) <= 2:
        print("请输入将要检测视频的文件路径和具体使用多少个线程参数！")
        print("示例(视频文件路径为\"./vlog.mp4\"，使用线程数为1)：python video.py \"./vlog.mp4\" 1")
        exit(-1)

    file_path = sys.argv[1]

    max_num_frame = 64
    max_inner_queue_size = 16
    num_processor = int(sys.argv[2])
    capture = None
    try :
        # 识别视频
        capture = cv2.VideoCapture(file_path)
    except:
        print(file_path + "：视频文件不存在！")
        exit(-1)

    frame_producer = FrameProducer("FrameProducer-0", max_num_frame, capture, num_processor)
    frame_processors = init_frame_processors(frame_producer,num_processor,max_inner_queue_size)
    frame_shower = FrameShower("FrameShower-0", frame_processors, (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    frame_producer.start()
    for processor in frame_processors:
        processor.start()
    frame_shower.start()