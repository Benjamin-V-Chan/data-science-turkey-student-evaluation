import pandas as pd
import os

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    df = df.copy()
    # rename for consistency
    df.rename(columns={
        'instr': 'instructor_id',
        'class': 'class_id',
        'nb.repeat': 'num_repeats'
    }, inplace=True)
    # ensure Q1–Q28 are within 1–5
    q_cols = [col for col in df.columns if col.startswith('Q')]
    valid = df[q_cols].applymap(lambda x: 1 <= x <= 5).all(axis=1)
    df = df[valid].reset_index(drop=True)
    return df

def save_data(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def main():
    raw_path = os.path.join('..', 'data', 'TurkiyeStudentEvaluation.csv')
    out_path = os.path.join('..', 'outputs', 'processed', 'processed_data.csv')
    df = load_data(raw_path)
    df_clean = preprocess_data(df)
    save_data(df_clean, out_path)

if __name__ == "__main__":
    main()
