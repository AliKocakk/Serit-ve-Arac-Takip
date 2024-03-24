from ultralytics import YOLO

# Load a model
model = YOLO("yolov8m.yaml")  # eğitilmiş paket oluşturma
model = YOLO("yolov8m.pt")  # hazır paketi eğitmek için kullanılıyor

# Use the model
model.train(data="coco128.yaml", epochs=3)  # videoyu coco128 veri görselleştirme ile eğitiliyor
metrics = model.val()  # validation yap
results = model("serit.mp4")  # videoyu tara
path = model.export(format="onnx")  # onnx derin öğrenme şeklinde kaydediliyor