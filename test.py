from models import YoloV5
predict_model = YoloV5(model_path="weights/yolov5s.pt",
                       weight_path="weights/best.pt",
                       image_path="test/images/AG-S-004_jpg.rf.dd3a1e229914fe956644912e1a857159.jpg")
result = predict_model.load_model()
