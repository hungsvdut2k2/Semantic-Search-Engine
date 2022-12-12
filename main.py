from fastapi import FastAPI, File, UploadFile
from fastapi.exceptions import HTTPException
from modules import YoloV5, SearchEngine, Post
import torch
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "DCMM Cau 0",
            "link": "https://colab.research.google.com/drive/1flYVHzi25el6rGBWARVM4fGO1E4VOsGb?usp=sharing"}


@app.post("/predict")
async def predict(file: UploadFile):
    if file.content_type != "image/jpeg":
        #raise HTTPException(400, detail="Invalid document type")
        return {"type": file.content_type}
    print(file)
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
async def search(search_query):
    posts = []
    search_engine = SearchEngine()
    search_engine.create_vocab()
    search_engine.create_docterm_matrix()
    vector_query = search_engine.vectorize(search_query)
    for result in search_engine.ranking(vector_query)[:10]:
        if search_query.lower() in result[2].lower():
            posts.append(result)
    return posts
