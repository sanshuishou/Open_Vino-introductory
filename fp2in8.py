from openvino.tools import mo
from openvino.runtime import serialize
import nncf
from api import *

#参数定义
image_size = 640
model_name = 'tphgf'
model_path = 'tphgf.onnx'
subset_size = 500
preset = nncf.QuantizationPreset.MIXED

#onnx转ir模型
fp32_path = f'xml_models/fp32_{model_name}.xml'
model = mo.convert_model(model_path)
serialize(model,fp32_path)

fp16_path = f'fp16_{model_name}.xml'
model = mo.convert_model(model_path,compress_to_fp16=True)
serialize(model,fp16_path)

#创造数据coco128
data_source = create_data_source('tphgf.yaml')
pot_data_loader = YOLOv5POTDataLoader(data_source)
#nncf量化数据
nncf_calibration_dataset = nncf.Dataset(data_source, transform_fn)
#创造量化算子
core = Core()
ov_model = core.read_model(fp32_path)
q_model = nncf.quantize(
    ov_model,nncf_calibration_dataset,preset=preset,subset_size=subset_size
)#开始nncf量化模型
nncf_int8_path = f'xml_models/{model_name}_nncf_int8.xml'
serialize(q_model,nncf_int8_path)



