from fastapi import FastAPI, File, UploadFile
from fastapi.exceptions import HTTPException
import torch
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict")
async def predict(file: UploadFile):
    if file.content_type != "image/jpeg":
        raise HTTPException(400, detail="Invalid document type")
    model = torch.hub.load(
        "ultralytics/yolov5", "custom", path="weights/best.pt"
    )
    img = "test/images/AG-S-004_jpg.rf.dd3a1e229914fe956644912e1a857159.jpg"
    final_results = []
    results = model(img)
    for i, row in results.pandas().xyxy[0].iterrows():
        xmin, ymin, xmax, ymax = (
            int(row["xmin"]),
            int(row["ymin"]),
            int(row["xmax"]),
            int(row["ymax"]),
        )
        confidence = float(row["confidence"])
        leaves_type = str(row["name"])
        final_results.append((leaves_type, confidence, xmin, ymin, xmax, ymax))
    return {"accuracy": final_results[0][1],
            "label": final_results[0][0],
            "xmin": final_results[0][2],
            "ymin": final_results[0][3],
            "xmax": final_results[0][4],
            "ymax": final_results[0][5]}


@app.get("/search/{search_query}")
async def search():
    return {"message:" "abc"}
