from fastapi import FastAPI, File, UploadFile
from fastapi.exceptions import HTTPException
from modules import YoloV5, SearchEngine, Post
import torch
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Dm Trang béo"}


@app.post("/predict")
async def predict(file: UploadFile):
    if file.content_type != "image/jpeg":
        raise HTTPException(400, detail="Invalid document type")
    img = "test/images/AG-S-004_jpg.rf.dd3a1e229914fe956644912e1a857159.jpg"
    yolov5 = YoloV5(weight_path="weights/best.pt", image_path=img)
    final_results = yolov5.predict_labels()
    return {"accuracy": final_results[0][1],
            "label": final_results[0][0],
            "xmin": final_results[0][2],
            "ymin": final_results[0][3],
            "xmax": final_results[0][4],
            "ymax": final_results[0][5]}


@app.get("/search/{search_query}")
async def search():
    return {"message:" "abc"}
