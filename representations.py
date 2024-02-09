import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from dataset import RudditDataset
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import gensim.downloader as api 
from enum import Enum


class EMBEDDING_TYPE(Enum):
        W2V_GOOGLE_300 = 1
        GLOVE_300 = 2
        FASTTEXT_300 = 3

class Representation:
    
    def __init__(self, data: pd.Series) -> None:
        self.data = data

    def bow(self):
        vectorizer = CountVectorizer()
        bow = vectorizer.fit_transform(self.data)
        return np.array(bow.toarray())
    
    def tfidf(self):
        vectorizer = TfidfVectorizer()
        tfidfs = vectorizer.fit_transform(self.data)
        return np.array(tfidfs.toarray())
    
    def w2vecs(self):
        corpus=[sent.split(" ") for sent in self.data]
        model = Word2Vec(corpus, vector_size=300, window=5, min_count=1, workers=2)
        return model
    
    def embeddings(self, model: EMBEDDING_TYPE = EMBEDDING_TYPE.W2V_GOOGLE_300, vector_size=300):
        embeddings_list = []
        print("Loading embedding model...")
        words_embeddings = Representation.embedding_model(model)
        print("Embedding model loaded")
        data_tokens = np.array([word_tokenize(sent) for sent in self.data])
        print("data_tokens", data_tokens.shape)
        for text_tokens in data_tokens:
            words_vecs = [words_embeddings[word] for word in text_tokens if word in words_embeddings]
            words_vecs = np.array(words_vecs) if len(words_vecs) > 0 else np.zeros(vector_size)
            embeddings_list.append(words_vecs.mean(axis=0))
        return np.array(embeddings_list)
    
    @staticmethod
    def glove_300_model():
        return api.load('glove-wiki-gigaword-300')
    
    @staticmethod
    def w2v_google_300_model():
        return api.load('word2vec-google-news-300')
    
    @staticmethod
    def fasttext_300_model():
        return api.load('fasttext-wiki-news-subwords-300')

    @staticmethod
    def pretrained_embeddings():
        return list(api.info()['models'].keys())

    @staticmethod
    def embedding_model(model: EMBEDDING_TYPE):
        if model == EMBEDDING_TYPE.W2V_GOOGLE_300:
            return Representation.w2v_google_300_model()
        elif model == EMBEDDING_TYPE.GLOVE_300:
            return Representation.glove_300_model()
        elif model == EMBEDDING_TYPE.FASTTEXT_300:
            return Representation.fasttext_300_model()
        else:
            raise ValueError("Invalid model")

if __name__ == '__main__':
    dataset = RudditDataset("data/ruddit.csv", cleaning=False)
    reps = Representation(dataset.train)
    print(reps.embeddings(EMBEDDING_TYPE.W2V_GOOGLE_300).shape)
    