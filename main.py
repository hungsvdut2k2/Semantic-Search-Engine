from fastapi import FastAPI, File, UploadFile
from fastapi.exceptions import HTTPException
from modules import SearchEngine, Post
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "DCMM Cau 0",
            "link": "https://colab.research.google.com/drive/1flYVHzi25el6rGBWARVM4fGO1E4VOsGb?usp=sharing"}


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
