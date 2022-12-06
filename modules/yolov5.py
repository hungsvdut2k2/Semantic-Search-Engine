import torch


class YoloV5:
    def __init__(self, weight_path, image_path, model_path="ultralytics/yolov5"):
        self.model_path = model_path
        self.weight_path = weight_path
        self.image_path = image_path

    def load_model(self):
        model = torch.hub.load(self.model_path, "custom", self.weight_path)
        return model

    def predict_labels(self):
        model = self.load_model()
        final_results = []
        results = model(self.image_path)
        for i, row in results.pandas().xyxy[0].iterrows():
            xmin, ymin, xmax, ymax = (
                int(row["xmin"]),
                int(row["ymin"]),
                int(row["xmax"]),
                int(row["ymax"]),
            )
            confidence = float(row["confidence"])
            leaves_type = str(row["name"])
            final_results.append(
                (leaves_type, confidence, xmin, ymin, xmax, ymax))
        return final_results
