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

def train_clustering(X_train, k=3):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_train)
    return km

def save_artifacts(models, reports):
    out_dir = os.path.join('..', 'outputs', 'models')
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(models['clf'],  os.path.join(out_dir, 'random_forest_classifier.joblib'))
    joblib.dump(models['kmeans'], os.path.join(out_dir, 'kmeans_clustering.joblib'))
    with open(os.path.join(out_dir, 'classification_report.json'), 'w') as f:
        json.dump(reports['classification'], f, indent=4)

def main():
    feat_dir = os.path.join('..', 'outputs', 'features')
    X_train, X_test, y_train, y_test = load_splits(feat_dir)
    clf   = train_classification_model(X_train, y_train)
    report = evaluate_classification(clf, X_test, y_test)
    kmeans = train_clustering(X_train, k=3)
    save_artifacts({'clf': clf, 'kmeans': kmeans}, {'classification': report})

if __name__ == "__main__":
    main()
