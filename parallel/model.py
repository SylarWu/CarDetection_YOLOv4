import threading
from queue import Queue
from yolo import YOLO
from utils.utils import letterbox_image,yolo_correct_boxes
import cv2
import numpy as np
import torch
from PIL import Image,ImageFont, ImageDraw
import time
import colorsys
from config.config import YOLOConfig

class Frame():

    def __init__(self,frame_index,frame_data,frame_shape,frame_data_preprocess,frame_detection = None):
        self._frame_index               = frame_index
        self._frame_data                = frame_data
        self._frame_data_preprocess     = frame_data_preprocess
        self._frame_detection           = frame_detection
        self._frame_shape               = frame_shape

    def getIndex(self):
        return self._frame_index

    def getData(self):
        return self._frame_data

    def getShape(self):
        return self._frame_shape

    def getPreprocessed(self):
        return self._frame_data_preprocess

    def getDetection(self):
        return self._frame_detection

    def setDetection(self, frame_detection):
        self._frame_detection = frame_detection


class FrameProducer(threading.Thread):
    def __init__(self, t_name, max_num_frame, capture, num_processors):
        threading.Thread.__init__(self, name=t_name)
        self.capture = capture
        self.num_processors = num_processors
        self.max_num_frame = max_num_frame
        self.waitToProcessQueue = Queue(max_num_frame)
        self.frame_index = 1
        self.config = YOLOConfig()
        self.stop = False
        self.totalNumFrames = capture.get(7)

    def run(self):
        while not self.stop:
            # 读取某一帧
            succ, frame = self.capture.read()

            if not succ and not self.isVideoEnd():
                # 视频帧未读取成功，但视频也并未结束
                continue

            if self.isVideoEnd():
                # 视频结束，给每个Processor线程发送结束消息
                for i in range(self.num_processors):
                    self.waitToProcessQueue.put(Frame(-1,None,None,None))
                self.stop = True
                break

            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))

            preprocessed, frame_shape = self.image_preprocess(frame)

            self.waitToProcessQueue.put(Frame(self.frame_index,frame,frame_shape,preprocessed))

            self.frame_index += 1
        self.capture.release()

    def image_numpyToTensor(self, image):
        image = np.array(image,dtype = np.float32)
        image /= 255.0
        image = np.transpose(image,(2,0,1))
        return image

    def image_preprocess(self,image):
        image_shape = np.array(np.shape(image)[0:2])

        crop_img = np.array(letterbox_image(image, (self.config.model_image_size[1], self.config.model_image_size[0])))

        photo = self.image_numpyToTensor(crop_img)
        images = np.asarray([photo])

        images = torch.from_numpy(images)
        if self.config.cuda:
            images = images.cuda()

        return images, image_shape

    def isVideoEnd(self):
        return self.frame_index >= self.totalNumFrames

    def getFrame(self):
        return self.waitToProcessQueue.get()

    def graceStop(self):
        self.stop = True

    def getStatus(self):
        return self.stop


class FrameProcessor(threading.Thread):
    def __init__(self, t_name, locka, lockb, max_inner_queue_size, producer):
        threading.Thread.__init__(self, name=t_name)
        self.yolo = YOLO()
        self.locka = locka
        self.lockb = lockb
        self.producer = producer
        self.max_inner_queue_size = max_inner_queue_size
        self.innerQueue = Queue(max_inner_queue_size)
        self.stop = False
        self.config = YOLOConfig()

    def run(self):
        while not self.stop:
            frames = []
            for _ in range(self.config.batch_size):
                self.locka.acquire()
                frame = self.producer.getFrame()
                self.lockb.release()
                if frame.getIndex() == -1:
                    break
                frames.append(frame)

            # batch_size=1时直接退出
            if len(frames) == 0:
                self.innerQueue.put(Frame(-1, None, None, None))
                self.stop = True
                break

            # batch_size>1时先把队列中还剩余的处理完，然后再退出
            preprocessed = torch.cat([_.getPreprocessed() for _ in frames],dim=0)
            detections = self.yolo.generate_detections(preprocessed)

            for _ in range(len(detections)):
                frames[_].setDetection(detections[_])
                self.innerQueue.put(frames[_])

            if len(frames) < self.config.batch_size:
                self.innerQueue.put(Frame(-1, None, None, None))
                self.stop = True
                break
        if not self.producer.getStatus():
            self.producer.graceStop()
            self.producer.getFrame()

    def getResult(self):
        return self.innerQueue.get()

    def graceStop(self):
        self.stop = True

    def getStatus(self):
        return self.stop


class FrameShower(threading.Thread):

    def __init__(self, t_name, frameprocessors, image_shape = None):
        threading.Thread.__init__(self, name=t_name)
        self.frameprocessors = frameprocessors
        self.yoloconfig = YOLOConfig()
        self.stop = False
        self.processor_status = [True for _ in range(len(self.frameprocessors))]
        self.vout = None

        if image_shape is not None:
            fps = 30
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

            vout = cv2.VideoWriter()
            vout.open('./output.mp4', fourcc, fps, image_shape, True)
            self.vout = vout

        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.yoloconfig.class_names), 1., 1.) for x in range(len(self.yoloconfig.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),self.colors))
        self.font = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * 1280 + 0.5).astype('int32'))

    def run(self):
        start = time.time()
        count = 0
        t1 = time.time()
        while not self.stop:

            for index, processor in enumerate(self.frameprocessors):
                if not self.processor_status[index]:
                    continue

                if count == 0:
                    t1 = time.time()
                frame = processor.getResult()
                count += 1

                if frame.getIndex() == -1:
                    self.processor_status[index] = False
                    continue

                data = self.postprocess(frame.getData(),frame.getShape(),frame.getDetection())

                data = np.array(data)
                # RGBtoBGR满足opencv显示格式
                data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

                self.vout.write(data)

                cv2.imshow("Video",data)

                t2 = time.time()
                if t2 - t1 >= 1.0:
                    print("FPS:%f" % (count / (t2 - t1)))
                    count = 0

                c = cv2.waitKey(20) & 0xff
                if c == 27:
                    self.stop = True
                    for index in range(len(self.processor_status)):
                        self.processor_status[index] = False
                    break

            if self.check_every_processor():
                self.stop = True
        end = time.time()
        print("Time usage : %fs" % (end - start))
        for processor in self.frameprocessors:
            if not processor.getStatus():
                processor.graceStop()
                for _ in range(self.yoloconfig.batch_size):
                    processor.getResult()

        self.vout.release()



    def postprocess(self, image, image_shape, detection):
        try:
            # 如果是None，代表未识别出物体，直接返回图像
            detection = detection.cpu().numpy()
        except:
            return image

        top_index = detection[:, 4] * detection[:, 5] > self.yoloconfig.confidence
        top_conf = detection[top_index, 4] * detection[top_index, 5]
        top_label = np.array(detection[top_index, -1], np.int32)
        top_bboxes = np.array(detection[top_index, :4])
        top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:, 0], -1), \
                                                 np.expand_dims(top_bboxes[:, 1], -1), \
                                                 np.expand_dims(top_bboxes[:, 2], -1), \
                                                 np.expand_dims(top_bboxes[:, 3], -1)

        # 去掉灰条
        boxes = yolo_correct_boxes(top_ymin,
                                   top_xmin,
                                   top_ymax,
                                   top_xmax,
                                   np.array([self.yoloconfig.model_image_size[0], self.yoloconfig.model_image_size[1]]),
                                   image_shape)

        thickness = (np.shape(image)[0] + np.shape(image)[1]) // self.yoloconfig.model_image_size[0]

        for i, c in enumerate(top_label):
            if c != 1 and c != 5 and c != 6 and c != 13 and c != 14:
                continue
            predicted_class = self.yoloconfig.class_names[c]
            score = top_conf[i]

            top, left, bottom, right = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, self.font)
            label = label.encode('utf-8')
            print(label)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[self.yoloconfig.class_names.index(predicted_class)])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[self.yoloconfig.class_names.index(predicted_class)])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=self.font)
            del draw
        return image

    def check_every_processor(self):
        stop = True
        for still_on in self.processor_status:
            if still_on:
                stop = False
        return stop
