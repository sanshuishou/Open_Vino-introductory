# Intel Open_Vino 实验文档

**author: @三水寿 😬**

## 一.实验环境条件

1)硬件要求：

Intel系列CPU/GPU 以下实验为CPU I7-1165G7

2)软件要求：

Window Liunx 皆可

![Untitled](Intel%20Open_Vino%20%E5%AE%9E%E9%AA%8C%E6%96%87%E6%A1%A3%2002179fc92eaa4c348a1860867da4f8be/Untitled.png)

![Untitled](Intel%20Open_Vino%20%E5%AE%9E%E9%AA%8C%E6%96%87%E6%A1%A3%2002179fc92eaa4c348a1860867da4f8be/Untitled%201.png)

Python 3.7以上

3)Open_Vino Python环境安装流程

Window & Liunx 相同

- python -m pip install --upgrade pip
- pip install openvino-dev==2023.0.1
- pip install nncf==2.5.0
- pip install torch==1.13.0+cpu torchvision==0.14.0+cpu torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cpu

等待pip安装完毕即可

PS：安装官网网站 [https://www.intel.cn/content/www/cn/zh/developer/tools/openvino-toolkit/download.html?ENVIRONMENT=DEV_TOOLS&OP_SYSTEM=LINUX&VERSION=v_2023_0_1&DISTRIBUTION=PIP](https://www.intel.cn/content/www/cn/zh/developer/tools/openvino-toolkit/download.html?ENVIRONMENT=DEV_TOOLS&OP_SYSTEM=LINUX&VERSION=v_2023_0_1&DISTRIBUTION=PIP)

## 二.模型转换及Nncf量化

PS：以下操作均基于Pytorch框架作为实例且以Yolov8作为实例展示。

1)模型转换办法

- Pytorch模型转换为Onnx模型
    
    ```python
    from ultralytics import YOLO
    model = YOLO('yolov8s.pt') 
    result = model.export(format='onnx') #yolov8原生转换
    ```
    
- onnx模型转vino模型（xml）
    
    ```python
    from openvino.tools import mo
    from openvino.runtime import serialize
    
    #model_path为onnx模型路径
    model = mo.convert_model(model_path)
    #fp32_parh为vino模型保存出来的路径
    serialize(model,fp32_path) #onnx2vino
    ```
    

2)Nncf模型量化

前提准备：Coco128数据集；     下方为下载地址：

[https://ultralytics.com/assets/coco128.zip](https://ultralytics.com/assets/coco128.zip)

```python
#不是coco任务的模型请使用自己的数据集 以下以coco举例

#创造数据coco128
data_source = create_data_source()
pot_data_loader = YOLOv5POTDataLoader(data_source)
#nncf量化数据 
nncf_calibration_dataset = nncf.Dataset(data_source, transform_fn)
#创造量化算子
core = Core()
#fp32_parh为vino模型保存出来的路径
ov_model = core.read_model(fp32_path)
q_model = nncf.quantize(
    ov_model,nncf_calibration_dataset,preset=preset,subset_size=subset_size
)#开始nncf量化模型
nncf_int8_path = f'{model_name}_nncf_int8.xml'
serialize(q_model,nncf_int8_path) #保存出量化后的xml模型
```

3)Val模型基准评估

- Val办法
    
    PS : api.py均来自Intel Open_Vino notebook内 本源码内带有
    
    ```python
    from api import *
    from ultralytics import YOLO
    from ultralytics.yolo.utils import DEFAULT_CFG
    from ultralytics.yolo.cfg import get_cfg
    from ultralytics.yolo.data.utils import check_det_dataset
    from openvino.runtime import Core
    from ultralytics.yolo.v8.detect.val import *
    
    NUM_TEST_SAMPLES = 700  #用多少张图片进行验证
    args = get_cfg(cfg=DEFAULT_CFG)
    args.data = str('tphgf.yaml') #自己的数据集
    DET_MODEL_NAME = "tph_gf"  #.前面的Name
    det_model = YOLO('模型路径')
    
    #分类情况
    label_map = {0:'dirt',1:'foliage',2:'guano',3:'feather'}
    
    #预测情况查看
    core = Core()
    det_ov_model = core.read_model('量化后的模型路径')
    device = 'CPU' #推理的设备
    det_compiled_model = core.compile_model(det_ov_model,device)
    #评估函数加载
    det_validator = DetectionValidator()
    det_validator.data = check_det_dataset(args.data)
    det_data_loader = det_validator.get_dataloader('数据目录', 1)
    
    #准确度评估
    #其中参数4为你自己的nc数量
    fp_det_stats = test(det_ov_model, core, det_data_loader, 4,det_validator, num_samples=NUM_TEST_SAMPLES)
    print_stats(fp_det_stats, det_validator.seen, det_validator.nt_per_class.sum())
    ```
    
- Benchmark评估
    
    ```powershell
    !benchmark_app -m 你的模型的路径 -d 你的硬件(CPU/GPU) -api async -shape "[1,3,640,640]”
    ```
    
- Yolov8s.pt基准表
    
    Latency :检测单个物体时的最小及最大延迟 （单位：ms）。
    
    mFPS:平衡帧率 from benchmark。
    
    mAP@:50 ：可以理解为在iou阈值为50时的平均准度。
    
    | Type | Latency(ms) | mFPS | mAP@:50 |
    | --- | --- | --- | --- |
    | FP32 | 188.84~403.97 | 13.18 | 0.772 |
    | FP16 | 193.24~403.89 | 13.4 | 0.772 |
    | Int8 | 58.28~116.94 | 43.87 | 0.744 |

4)量化后模型推理办法

```python
from api import *
from openvino.runtime import Core
import cv2
import time
from PIL import Image
import numpy as np

def predict(model,cls_maps:dict,obj_path:str,cap=False):
    """
    使用Open_Vino量化后预测(目标检测）
    :param model: xml模型（量化后的模型）
    :param cls_maps: 输入dict类型，{编号:对应标签}
    :param obj_path: 图片或视频路径，设置为'0'且cap为True的时候使用摄像头
    :param cap: 是否使用摄像头,bool类型
    :return: 图片模式返回预测后的图片，视频模式返回预测后的视频，摄像头模式返回摄像头录制的视频
    """
    label_map = cls_maps
    #使用xml进行预测
    core = Core()
    det_ov_model = core.read_model(model)
    device = 'CPU'
    det_compiled_model = core.compile_model(det_ov_model, device)

    if obj_path.split('.')[-1] == 'mp4':
        print('视频模式')
        cm = cv2.VideoCapture(obj_path)
        while True:
            a,frame = cm.read()
            if a:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = detect(frame, det_compiled_model,nc=4)[0]
                image_with_boxes = draw_results(detections, frame, label_map)
                image_with_boxes = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
                cv2.imshow('vid',image_with_boxes)
                if cv2.waitKey(1) & 0xff==ord('q'):
                    break
                else:
                    continue
            else:
                print('mp4未打开成功！')
                break
    elif obj_path == '0' and cap:
        print('摄像头模式')
        cm = cv2.VideoCapture(4)
        while True:
            a,frame = cm.read()
            if a:
                t1 = time.time()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = detect(frame, det_compiled_model,4)[0]
                image_with_boxes = draw_results(detections, frame, label_map)
                t2 = time.time()
                ms = int((t2-t1)*1000)
                cv2.putText(image_with_boxes,f'FPS:{1000/ms}',(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,255,0),2)
                image_with_boxes = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
                cv2.imshow('cm',image_with_boxes)
                if cv2.waitKey(1)&0xff==ord('q'):
                    break
                else:
                    continue
            else:
                print('摄像头未打开成功！')
                break
    else:
        print('图片模式或其他')
        frame = Image.open(obj_path)
        frame = np.array(frame)
        detections = detect(frame, det_compiled_model,4)[0]
        image_with_boxes = draw_results(detections, frame, label_map)
        # cv2.imshow('images', image_with_boxes)
        # cv2.waitKey(0)
        img = Image.fromarray(image_with_boxes)
        img.show()
        # cv2.imwrite('result/001.jpg',image_with_boxes)
```