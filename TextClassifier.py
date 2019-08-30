# Word tokenization
import pandas as pd
import string
import spacy
import numpy as np
from spacy.lang.en import English
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn import metrics
import warnings

# Loading data
warnings.filterwarnings("ignore", category=FutureWarning)
with open("trainingSet.json", 'r') as f:
    dt = eval(f.read())
dataset = pd.DataFrame(np.array(dt))
punctuations = string.punctuation

nlp = spacy.load('en')
# Create our list of stopwords
stop_words = spacy.lang.en.stop_words.STOP_WORDS

parser = English()

def spacy_tokenizer(sentence):   
    mytokens = parser(sentence)
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]
    return mytokens

class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

def clean_text(text):
    # Removing spaces and converting text into lowercase    
    return text.strip().lower()

bow_vector = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))
tfidf_vector = TfidfVectorizer(tokenizer = spacy_tokenizer)


data = dataset[0] # the features
labels = dataset[1] # the labels
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)

classifier = LogisticRegression()
# Create pipeline using Bag of Words
pipe = Pipeline([("cleaner", predictors()),
                 ('vectorizer', bow_vector),
                 ('classifier', classifier)])

# model generation
pipe.fit(X_train,y_train)

# Predicting with a test datasetS
predicted = pipe.predict(X_test)

# Model Accuracy
print("Logistic Regression Accuracy:",metrics.accuracy_score(y_test, predicted))
print("Logistic Regression Precision:",metrics.precision_score(y_test, predicted,average='micro'))
print("Logistic Regression Recall:",metrics.recall_score(y_test, predicted,average='micro'))
