import numpy as np


def euclidean_distance(vector_a, vector_b):
    return np.linalg.norm(vector_a - vector_b)


def cosine_similarity(vector_a, vector_b):
    np_vector_a = np.array(vector_a)
    np_vector_b = np.array(vector_b)
    return np.dot(np_vector_a, np_vector_b)/(np.linalg.norm(np_vector_a) * np.linalg.norm(np_vector_b))
