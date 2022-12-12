import nltk
import string
import numpy as np
from nltk.corpus import stopwords
nltk.download('stopwords')


def remove_punctuations(text: str) -> str:
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_stopwords(text: str) -> str:
    for word in stopwords.words('english'):
        new_text = text
        if word in new_text:
            new_text = new_text.replace(word, '')
    return new_text


def normalize_text(text: str) -> str:
    normalized_text = text.lower()
    normalized_text = remove_punctuations(normalized_text)
    normalized_text = remove_stopwords(normalized_text)
    return normalized_text
