from api import *
import numpy as np
from openvino.runtime import Core
from ultralytics.yolo.utils import DEFAULT_CFG
from ultralytics.yolo.cfg import get_cfg
import cv2
import time

label_map = {0:'person'}
#使用xml进行预测
core = Core()
det_ov_model = core.read_model('yolov8s_nncf_int8.xml')
device = 'CPU'
path = '/home/vkrobot/Corleone Ge/Openvino/data/images/predict/bb_V0016_I0001600.jpg'

t1 = time.time()
det_compiled_model = core.compile_model(det_ov_model,device)
print(f'1耗时:{time.time() - t1}')
t2 = time.time()
img0 = cv2.imread(path)
print(f'2耗时:{time.time() - t2}')
t3 = time.time()
img = letterbox(img0)[0]
print(f'3耗时:{time.time() - t3}')
t4 = time.time()
img = img.transpose(2, 0, 1)
print(f'5耗时:{time.time() - t4}')
t5 = time.time()
img = np.ascontiguousarray(img)
print(f'6耗时:{time.time() - t5}')
t6 = time.time()
img = np.expand_dims(img,0)
print(f'7耗时:{time.time() - t6}')
t7 = time.time()
detections = det_compiled_model(img)
print(f'8耗时:{time.time() - t7}')

print(detections)
# image_with_boxes = draw_results(detections, frame, label_map)
# cv2.imshow('vid',image_with_boxes)