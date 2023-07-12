from ultralytics import YOLO

model = YOLO('pt_models/tph_gf.pt')
model.export(format='onnx')