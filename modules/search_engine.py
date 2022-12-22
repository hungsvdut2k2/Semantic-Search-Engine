from utils import cosine_similarity, euclidean_distance
import numpy as np
import pandas as pd
from gensim.models import Word2Vec


class SearchEngine:
    def __init__(self) -> None:
        self.model = Word2Vec.load(
            "weights/word2vec.model")
