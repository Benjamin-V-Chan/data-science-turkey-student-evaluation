import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path):
    return pd.read_csv(path)

def create_thematic_scores(df):
    df = df.copy()
    content = [f'Q{i}' for i in range(1, 5)]
    delivery = [f'Q{i}' for i in range(5, 11)]
    performance = [f'Q{i}' for i in range(11, 29)]
    df['score_content'] = df[content].mean(axis=1)
    df['score_delivery'] = df[delivery].mean(axis=1)
    df['score_performance'] = df[performance].mean(axis=1)
    return df

def encode_and_scale(df):
    features = ['score_content', 'score_delivery', 'score_performance', 'attendance', 'num_repeats']
    X_num = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)
    X = pd.DataFrame(X_scaled, columns=features)
    y = df['difficulty']
    return X, y

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def save_splits(X_train, X_test, y_train, y_test):
    out_dir = os.path.join('..', 'outputs', 'features')
    os.makedirs(out_dir, exist_ok=True)
    X_train.to_csv(os.path.join(out_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(out_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(out_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(out_dir, 'y_test.csv'), index=False)

def main():
    df = load_data(os.path.join('..', 'outputs', 'processed', 'processed_data.csv'))
    df = create_thematic_scores(df)
    X, y = encode_and_scale(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    save_splits(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
