import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
import joblib
import json

def load_splits(path):
    X_train = pd.read_csv(os.path.join(path, 'X_train.csv'))
    X_test  = pd.read_csv(os.path.join(path, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(path, 'y_train.csv')).squeeze()
    y_test  = pd.read_csv(os.path.join(path, 'y_test.csv')).squeeze()
    return X_train, X_test, y_train, y_test

def train_classification_model(X_train, y_train):
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    return clf

def evaluate_classification(clf, X_test, y_test):
    report = classification_report(y_test, clf.predict(X_test), output_dict=True)
    return report

