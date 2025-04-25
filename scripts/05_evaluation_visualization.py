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

def plot_roc_curve(clf, X_test, y_test, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    y_score = clf.predict_proba(X_test)
    classes = clf.classes_
    for idx, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test == cls, y_score[:, idx])
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0,1], [0,1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for class {cls}')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'roc_curve_class_{cls}.png'))
        plt.close()

def plot_clusters(kmeans, X, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    pca = PCA(n_components=2)
    comps = pca.fit_transform(X)
    plt.figure()
    plt.scatter(comps[:,0], comps[:,1], c=kmeans.labels_, cmap='viridis')
    plt.title('KMeans Clusters (PCA Projection)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'clusters_pca.png'))
    plt.close()

