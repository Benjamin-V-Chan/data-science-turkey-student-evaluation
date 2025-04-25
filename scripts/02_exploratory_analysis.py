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

def plot_correlations(df, cols, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    corr = df[cols].corr()
    plt.figure(figsize=(12, 10))
    plt.imshow(corr, aspect='auto', cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.xticks(range(len(cols)), cols, rotation=90)
    plt.yticks(range(len(cols)), cols)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()

def main():
    data_path = os.path.join('..', 'outputs', 'processed', 'processed_data.csv')
    fig_dir = os.path.join('..', 'outputs', 'figures')
    df = load_data(data_path)
    q_cols = [col for col in df.columns if col.startswith('Q')]
    cat_cols = ['attendance', 'difficulty']
    plot_distribution(df, q_cols + cat_cols, fig_dir)
    plot_correlations(df, q_cols + cat_cols, fig_dir)

if __name__ == "__main__":
    main()
