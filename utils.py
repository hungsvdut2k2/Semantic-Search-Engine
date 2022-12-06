import numpy as np


def vector_search(query, model, index, num_results=10):
    vector = model.encode(list(query))
    D, I = index.search(np.array(vector).astype("float32"), k=num_results)
    return D, I


def id2details(df, I, column):
    return [list(df[df.id == idx][column]) for idx in I[0]]
