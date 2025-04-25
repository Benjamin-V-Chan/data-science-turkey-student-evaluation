import pandas as pd
import matplotlib.pyplot as plt
import os

def load_data(path):
    return pd.read_csv(path)

def plot_distribution(df, cols, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for col in cols:
        plt.figure()
        df[col].hist(bins=5)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{col}_distribution.png'))
        plt.close()

