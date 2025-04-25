# 1. Import pandas, os
# 2. Define load_data(path) → reads CSV
# 3. Define preprocess_data(df):
#      - rename columns instr→instructor_id, class→class_id, nb.repeat→num_repeats
#      - verify Q1–Q28 ∈ [1,5]; handle missing if any
# 4. Define save_data(df, path) → create dirs & write CSV
# 5. main():
#      - raw_path = '../data/TurkiyeStudentEvaluation.csv'
#      - out_path = '../outputs/processed/processed_data.csv'
#      - df = load_data(raw_path)
#      - df_clean = preprocess_data(df)
#      - save_data(df_clean, out_path)
