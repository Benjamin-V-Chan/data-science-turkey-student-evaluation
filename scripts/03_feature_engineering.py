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

