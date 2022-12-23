from utils import cosine_similarity, euclidean_distance
import numpy as np
import pandas as pd
from gensim.models import Word2Vec


class SearchEngine:
    def __init__(self) -> None:
        self.model = Word2Vec.load(
            "weights/word2vec.model")
        self.docterm_matrix = []

    def create_docterm_matrix(self):
        df = pd.read_csv('dataset/output.csv')
        for i in range(len(df)):
            temp_list = []
            count = 0
            vector = np.zeros(100)
            temp_list.append(i)
            temp_list.append(df.loc[i]['title'])
            temp_list.append(df.loc[i]['article'])
            for word in df.loc[i]['article'].split():
                try:
                    vector += self.model.wv[word]
                    count += 1
                except:
                    print("error")
                    vector = vector / count
            temp_list.append(vector)
            self.docterm_matrix.append(temp_list)

    def query_embedding(self, query):
        query_embedding = np.zeros(100)
        for word in query.lower().split():
            query_embedding += self.model.wv[word]
        return query_embedding

    def search(self, query):
        results = []
        final_results = []
        query_embedding = self.query_embedding(query)
        for corpus in self.docterm_matrix:
            temp_list = []
            temp_list.append(corpus[0])
            temp_list.append(corpus[1])
            temp_list.append(corpus[2])
            cosine_value = cosine_similarity(query_embedding, corpus[3])
            temp_list.append(cosine_value)
            results.append(temp_list)
        results.sort(reverse=True, key=lambda x: x[3])
        for corpus in results:
            if query.lower() in corpus[2]:
                final_results.append((corpus[0], corpus[1]))
        return final_results

    def get_corpus(self, index):
        df = pd.read_csv("dataset/output.csv")
        return (df.loc[index]['title'], df.loc[index]['article'])
