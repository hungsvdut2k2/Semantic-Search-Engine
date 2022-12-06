from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/search/{search_query}")
async def search():
    return {"message:" "abc"}
