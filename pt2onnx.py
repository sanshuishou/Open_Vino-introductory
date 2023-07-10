from ultralytics import YOLO

model = YOLO('yolov8s.pt')
result = model.export(format='onnx')