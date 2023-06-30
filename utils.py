import string
import numpy as np
from sklearn import pipeline, linear_model, model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def split_text(text: str) -> list[str]:
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    return text.split(" ")

def get_word_counts(words: list[str], keeptop=None):
    word_counts = {}

    for word in words:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1

    if "" in word_counts.keys(): del word_counts[""]

    if keeptop is not None:
        word_counts = dict(sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:keeptop])
    

    return word_counts

def prepare_data(df, train_size=0.7):
    embeddings = df.filter(regex=r'^e\d+')
    ratings = df["Rating"][embeddings.notna().all(axis=1)]
    embeddings = embeddings.dropna()

    assert ratings.shape[0] == embeddings.shape[0]

    X_train, X_test, y_train, y_test = [x.values for x in train_test_split(embeddings, ratings, train_size=train_size, stratify=ratings)]

    return X_train, X_test, y_train, y_test

def get_cv():
    regressor = linear_model.Ridge()
    model = pipeline.Pipeline([("standardscaler", StandardScaler()), ("regressor", regressor)])
    param_grid = {
        "regressor__alpha": [10**i for i in np.linspace(2, 4, 10)]
    }
    cv = model_selection.GridSearchCV(model, param_grid, cv=5, verbose=3, scoring="neg_mean_squared_error")

    return cv
