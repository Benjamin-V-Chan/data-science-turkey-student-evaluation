# 1. Import pandas, os, sklearn.ensemble, sklearn.cluster, sklearn.metrics, joblib, json
# 2. Define load_splits(path) → read X_train, X_test, y_train, y_test
# 3. Define train_classification_model(X_train,y_train) → RandomForestClassifier.fit
# 4. Define evaluate_classification(clf,X_test,y_test) → classification_report(output_dict=True)
# 5. Define train_clustering(X_train,k=3) → KMeans.fit
# 6. Define save_artifacts(models, reports):
#      - mkdir ../outputs/models
#      - joblib.dump classifiers & clustering
#      - json.dump classification_report
# 7. main(): chain above
