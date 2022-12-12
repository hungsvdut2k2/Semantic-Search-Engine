import io
from utils import remove_punctuations, remove_stopwords, normalize_text
import numpy as np
import os


class SearchEngine:
    def __init__(self) -> None:
        self.vocab = []
        self.docterm_matrix = {}
        self.doc_lists = []

    def create_vocab(self):
        file_name = os.listdir('data/Data')
        file_name.remove('.git')
        root_folder = 'data/Data'
        for folder in file_name:
            folder_path = os.path.join(root_folder, folder)
            for corpus in os.scandir(folder_path):
                with io.open(corpus, 'rb') as f:
                    lines = f.readlines()
                    title = lines[0].strip()
                    article = b' '.join(lines[1:]).strip()
                    article = article.decode('utf-8')
                    article = normalize_text(article)
                    if (title, article) not in self.doc_lists:
                        self.doc_lists.append((title, article))
                    tokens = article.split(' ')
                    for token in tokens:
                        if token not in self.vocab:
                            self.vocab.append(token)

    def vectorize(self, article):
        res = []
        text = normalize_text(article)
        for word in self.vocab:
            res.append(text.count(word))
        return res

    def create_docterm_matrix(self):
        for (title, article) in self.doc_lists:
            vector = self.vectorize(article)
            self.docterm_matrix[title, article] = vector
        return self.docterm_matrix

    def calculate_cosine_similarity(self, vector_a, vector_b):
        np_vector_a = np.array(vector_a)
        np_vector_b = np.array(vector_b)
        return np.dot(np_vector_a, np_vector_b)/(np.linalg.norm(np_vector_a) * np.linalg.norm(np_vector_b))

    def ranking(self, vector_query):
        ranking = []
        for doc_info, vector in self.docterm_matrix.items():
            cosine_similarity = self.calculate_cosine_similarity(
                vector_query, vector)
            ranking.append((cosine_similarity, doc_info[0], doc_info[1]))
        return sorted(ranking, reverse=True)
