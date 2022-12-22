from fastapi import FastAPI, File, UploadFile
from fastapi.exceptions import HTTPException
from modules import SearchEngine
from gensim.models import Word2Vec
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "DCMM Cau 0",
            "link": "https://colab.research.google.com/drive/1flYVHzi25el6rGBWARVM4fGO1E4VOsGb?usp=sharing"}


@app.get("/search/{search_query}")
async def search(search_query):
    return ""
