# 1. Import pandas, os, sklearn.model_selection, sklearn.preprocessing
# 2. Define load_data(path)
# 3. Define create_thematic_scores(df):
#      - content = mean(Q1–Q4), delivery = mean(Q5–Q10), performance = mean(Q11–Q28)
# 4. Define encode_and_scale(df):
#      - select numeric features [scores, attendance, num_repeats]
#      - StandardScaler → X_scaled, y = difficulty
# 5. Define split_data(X,y) → train_test_split(test_size=0.2,stratify=y)
# 6. Define save_splits(...) → mkdir ../outputs/features + write X_train.csv etc.
# 7. main(): chain above
