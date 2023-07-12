from api import *
from ultralytics import YOLO
from ultralytics.yolo.utils import DEFAULT_CFG
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.data.utils import check_det_dataset
from openvino.runtime import Core
from ultralytics.yolo.v8.detect.val import *
#超参数
NUM_TEST_SAMPLES = 700
args = get_cfg(cfg=DEFAULT_CFG)
args.data = str('coco128.yaml')
DET_MODEL_NAME = "yolov8s"
det_model = YOLO('pt_models/yolov8s.pt')
label_map = det_model.model.names

#预测情况查看
# res = det_model(IMAGE_PATH)
# Image.fromarray(res[0].plot()[:, :, ::-1])
core = Core()
det_ov_model = core.read_model('xml_models/yolov8s_nncf_int8.xml')
device = 'CPU'
det_compiled_model = core.compile_model(det_ov_model,device)
#评估函数加载
det_validator = DetectionValidator()
det_validator.data = check_det_dataset(args.data)
det_data_loader = det_validator.get_dataloader('coco128/images/', 1)

#准确度评估
fp_det_stats = test(det_ov_model, core, det_data_loader, 80,det_validator, num_samples=NUM_TEST_SAMPLES)
print_stats(fp_det_stats, det_validator.seen, det_validator.nt_per_class.sum())