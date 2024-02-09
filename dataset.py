import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
nltk.download('stopwords')
nltk.download('wordnet')

class RudditDataset():

    def __init__(self, path=None, standardize=True, down_sample=True, cleaning=True) -> None:
        self.path = path
        self.train = None
        self.test = None
        self.train_labels = None
        self.test_labels = None
        self.down_sampled = down_sample
        print(f"Loading data from {path}")
        self.data = self.load(path) if path else None
        if standardize:
            print("Standardizing data")
            self.data = self.standardize()
        if down_sample:
            print("Down sampling data")
            self.down_sample()
        if cleaning:
            print("Cleaning data")
            self.cleaning()

    def load(self, path: str) -> pd.DataFrame:
        self.data = pd.read_csv(path)
        return self.data
    
    def standardize(self, df: pd.DataFrame=None) -> pd.DataFrame:
        if df is None:
            df = self.data
        df = df[['comment_text', 'offensiveness_score']]
        df = df.rename(columns={'comment_text': 'text', 'offensiveness_score': 'score'})
        return df
    
    def down_sample(self, df: pd.DataFrame=None, train_size= 0.75, test_size=0.25, test_valid_ratio=0.5):
        if df is None:
            df = self.data
        x_train, x_test_valid, y_train, y_test_valid = train_test_split(df['text'], df['score'], train_size=train_size, test_size=test_size, random_state=0)
        x_test, x_valid, y_test, y_valid = train_test_split(x_test_valid, y_test_valid, test_size=0.5, random_state=0)
        self.train = x_train.reset_index(drop=True)
        self.test = x_test.reset_index(drop=True)
        self.valid = x_valid.reset_index(drop=True)
        self.train_labels = y_train.reset_index(drop=True)
        self.test_labels = y_test.reset_index(drop=True)
        self.valid_labels = y_valid.reset_index(drop=True)

        return self.train, self.train_labels, self.test, self.test_labels, self.valid, self.valid_labels
    
    def clean_data(self, corpus: pd.Series, lower_case=True, remove_punctuation=True, remove_stopwords=True, lemmatize=True):
        cleaned_docs = []
        for doc in corpus:
            text_data = doc
            if remove_punctuation:
                text_data = re.sub('[^a-zA-Z]', ' ', doc)
            if lower_case:
                text_data = text_data.lower()
            if remove_stopwords:
                text_data = ' '.join([word for word in text_data.split() if not word in set(stopwords.words('english'))])
            if lemmatize:
                wl = WordNetLemmatizer()
                text_data = ' '.join([wl.lemmatize(word) for word in text_data.split()])

            cleaned_docs.append(text_data)
        return pd.Series(cleaned_docs)
    
    def cleaning(self):
        self.train = self.clean_data(self.train)
        self.test = self.clean_data(self.test)
        self.valid = self.clean_data(self.valid)

        return self.train, self.test, self.valid
    
    def splited(self, data_type="train"):
        if data_type == 'train':
            data = self.train
        elif data_type == 'test':
            data = self.test
        elif data_type == 'valid':
            data = self.valid
        
        return np.array([sent.split(" ") for sent in data])
    
    @staticmethod
    def vocabulary(data):
        vectorizer = CountVectorizer()
        vectorizer.fit_transform(data)
        return vectorizer.get_feature_names()
    
    @staticmethod
    def tokenized(data):
        return np.array([word_tokenize(sent) for sent in data])
    
    def __repr__(self) -> str:
        if self.down_sampled:
            return f"RudditDataset(train={self.train.shape}, test={self.test.shape})"
        return f"RudditDataset(data={self.data.shape})"

if __name__ == "__main__":
    dataset = RudditDataset("data/ruddit.csv")