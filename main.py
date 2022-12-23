from fastapi import FastAPI, File, UploadFile
from fastapi.exceptions import HTTPException
from modules import SearchEngine
app = FastAPI()
search_engine = SearchEngine()
search_engine.create_docterm_matrix()


@app.get("/")
async def root():
    return {"message": "DCMM Cau 0",
            "link": "https://colab.research.google.com/drive/1flYVHzi25el6rGBWARVM4fGO1E4VOsGb?usp=sharing"}


@app.get("/search/{search_query}")
async def search(search_query):
    return search_engine.search(search_query)


@app.get("/corpus/{corpus_index}")
async def get_corpus(corpus_index):
    return search_engine.get_corpus(int(corpus_index))
