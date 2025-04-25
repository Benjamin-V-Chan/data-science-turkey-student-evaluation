# 1. Import pandas, os, matplotlib.pyplot, sklearn.metrics, sklearn.decomposition, joblib, json
# 2. Define load_models(path) → load clf, kmeans, report json
# 3. Define load_test_data(path) → read X_test, y_test
# 4. Define plot_confusion_matrix(...)
# 5. Define plot_roc_curve(...)
# 6. Define plot_clusters(kmeans, X, output_dir):
#      - PCA(n_components=2) → scatter colored by kmeans.labels_
# 7. Define generate_report(report_data, output_file) → write Markdown with embedded JSON
# 8. main(): orchestrate all steps, saving figs to ../outputs/figures and report to ../outputs/report.md
