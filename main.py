from fastapi import FastAPI, File, UploadFile

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict")
async def predict(file: UploadFile):
    return {"filename": file.filename}


@app.get("/search/{search_query}")
async def search():
    return {"message:" "abc"}
