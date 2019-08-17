# Imports
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def calculate(message):
    # Create data frame from yelp reviews
    dataset = pd.read_csv('/app/python/yelp_labelled.txt', sep='\t', header=None)
    dataset.columns = ['Text', 'Class']

    train_X = dataset['Text'].values
    train_y = dataset['Class'].values

    # Vectorizer to represent our data
    vectorizer = TfidfVectorizer()
    vectorized_train_X = vectorizer.fit_transform(train_X)

    # Create Logistic Regression model
    classifier = LogisticRegression()
    classifier.fit(vectorized_train_X, train_y)

    # Use test message and convert to vector representation
    sentence = [message]
    vectorized_message = vectorizer.transform(sentence)
    prediction = classifier.predict_proba(vectorized_message)

    # Get probabilites
    negative_score = prediction[0][0]
    positive_score = prediction[0][1]

    # Determine negative, positive, neutral
    if negative_score >= .33 and positive_score >= .33:
        sentiment = 'neutral'
    elif negative_score > .66:
        sentiment = 'negative'
    else:
        sentiment = 'positive'

    return sentiment
