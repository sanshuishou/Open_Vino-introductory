from PIL import Image
from api import *
from ultralytics import YOLO
from ultralytics.yolo.utils import DEFAULT_CFG
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.data.utils import check_det_dataset
from openvino.runtime import Core
import numpy as np
import os

path = 'data/Gf/images/train/'
files = os.listdir(path)
label_map = {0:'dirt',1:'foliage',2:'guano',3:'fqeather'}
#使用xml进行预测
core = Core()
det_ov_model = core.read_model('/home/vkrobot/Corleone_Ge/Openvino/fp16_tphgf.xml')
device = 'CPU'
det_compiled_model = core.compile_model(det_ov_model, device)
for i in files:
    p = path+i
    img = Image.open(p)
    img_arr = np.array(img)
    detections = detect(img_arr, det_compiled_model)[0]
    image_with_boxes = draw_results(detections, img_arr, label_map)
    # cv2.imshow('images', image_with_boxes)
    # cv2.waitKey(0)
    img = Image.fromarray(image_with_boxes)
    img.save('result/'+i)