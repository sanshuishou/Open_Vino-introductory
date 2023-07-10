from PIL import Image
from api import *
from ultralytics import YOLO
from ultralytics.yolo.utils import DEFAULT_CFG
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.data.utils import check_det_dataset
from openvino.runtime import Core

#超参数
NUM_TEST_SAMPLES = 300
args = get_cfg(cfg=DEFAULT_CFG)
args.data = str('miniobj.yaml')
DET_MODEL_NAME = "yolov8s"
IMAGE_PATH = '/home/vkrobot/Corleone Ge/Openvino/data/images/predict/bb_V0016_I0001600.jpg'
det_model = YOLO('/home/vkrobot/Corleone Ge/Openvino/miniv8ntphv1.pt')
label_map = det_model.model.names

#预测情况查看
res = det_model(IMAGE_PATH)
Image.fromarray(res[0].plot()[:, :, ::-1])
core = Core()
det_ov_model = core.read_model('yolov8ntphv1_nncf_int8.xml')
device = 'CPU'
det_compiled_model = core.compile_model(det_ov_model,device)
input_image = np.array(Image.open(IMAGE_PATH))
detections = detect(input_image, det_compiled_model)[0]
image_with_boxes = draw_results(detections, input_image, label_map)

img = Image.fromarray(image_with_boxes)
Image.Image.save(img,'result/001.jpg')

#评估函数加载
det_validator = det_model.ValidatorClass(args=args)
det_validator.data = check_det_dataset(args.data)
det_data_loader = det_validator.get_dataloader('/home/vkrobot/Corleone Ge/Openvino/data/images/predict/', 1)
det_validator.is_coco = True
det_validator.class_map = ops.coco80_to_coco91_class()
det_validator.names = det_model.model.names
det_validator.metrics.names = det_validator.names
det_validator.nc = det_model.model.model[-1].nc

#准确度评估
fp_det_stats = test(det_ov_model, core, det_data_loader, det_validator, num_samples=NUM_TEST_SAMPLES)
print_stats(fp_det_stats, det_validator.seen, det_validator.nt_per_class.sum())