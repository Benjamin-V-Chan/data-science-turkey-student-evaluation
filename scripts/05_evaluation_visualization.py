import pandas as pd
import os
import matplotlib.pyplot as plt
import joblib
import json
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc
from sklearn.decomposition import PCA

def load_models(path):
    clf = joblib.load(os.path.join(path, 'random_forest_classifier.joblib'))
    kmeans = joblib.load(os.path.join(path, 'kmeans_clustering.joblib'))
    report = json.load(open(os.path.join(path, 'classification_report.json')))
    return clf, kmeans, report

def load_test_data(path):
    X_test = pd.read_csv(os.path.join(path, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(path, 'y_test.csv')).squeeze()
    return X_test, y_test

def plot_confusion_matrix(clf, X_test, y_test, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    disp = ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, normalize='true')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

