import numpy as np
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from nltk.stem import PorterStemmer


def load_data(data_path, columns):
    df = pd.read_csv(data_path, encoding='latin')
    df = df[columns]
    df.drop_duplicates(inplace=True)
    df = df.dropna()
    return df

def process_text(text):

    text = text.lower()
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    nostopword = [word for word in nopunc.split() if word.lower() not in stopwords]
    stemmed_text = [stemmer.stem(word) for word in nostopword]

    return stemmed_text

def multinominalNaiveBayes(X_train, y_train):
    classifier = MultinomialNB().fit(X_train, y_train)
    prediction = classifier.predict(X_train)
    print(classification_report(y_train, prediction))
    print(confusion_matrix(y_train, prediction))

def logisticRegression(X_train, y_train):
    classifier = LogisticRegression().fit(X_train, y_train)
    prediction = classifier.predict(X_train)
    print(classification_report(y_train, prediction))
    print(confusion_matrix(y_train, prediction))



if __name__ == "__main__":
    # Import datasets, drop duplicate and null rows
    df = load_data('email_data_path', ['text', 'spam'])
    df2 = load_data('spam_data_path', ['value', 'sentence'])

    stopwords = stopwords.words('english')
    stemmer = PorterStemmer()

    # Bag of words for feature extraction and preprocessing
    sentences_bow = CountVectorizer(analyzer=process_text).fit_transform(df['text'])
    sentences_bow2 = CountVectorizer(analyzer=process_text).fit_transform(df2['sentence'])

    # Split data and set variables for testing
    X_train, X_test, y_train, y_test = train_test_split(sentences_bow, df['spam'], test_size=0.20, random_state=200)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(sentences_bow2, df2['value'], test_size=0.20, random_state=200)
    #sentences_bow.shape
    #sentences_bow2.shape

    # Prediction with training data for multinomial naive bayes
    print("Multinomial Naive Bayes on Dataset 1:")
    multinominalNaiveBayes(X_train, y_train)

    print("Multinomial Naive Bayes on Dataset 2:")
    multinominalNaiveBayes(X_train2, y_train2)

    # Prediction with test data for multinomial naive bayes on dataset 2
    print("Multinomial Naive Bayes on Dataset 1 test data:")
    multinominalNaiveBayes(X_test, y_test)

    print("Multinomial Naive Bayes on Dataset 2 test data:")
    multinominalNaiveBayes(X_test2, y_test2)


    # Prediction with training data for logistic regression
    print("Logistic Regression on Dataset 1:")
    logisticRegression(X_train, y_train)

    print("Logistic Regression on Dataset 2:")
    logisticRegression(X_train2, y_train2)

    # Prediction with test data for logistic regression on dataset 2
    print("Logistic Regression on Dataset 1 test data:")
    logisticRegression(X_test, y_test)

    print("Logistic Regression on Dataset 2 test data:")
    logisticRegression(X_test2, y_test2)