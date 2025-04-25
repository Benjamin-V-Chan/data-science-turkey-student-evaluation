# 1. Import pandas, matplotlib.pyplot, os
# 2. Define load_data(path) → reads cleaned CSV
# 3. Define plot_distribution(df, cols, output_dir):
#      - for each col: histogram → savefig
# 4. Define plot_correlations(df, cols, output_dir):
#      - compute df[cols].corr()
#      - draw heatmap via plt.imshow → savefig
# 5. main():
#      - data_path = '../outputs/processed/processed_data.csv'
#      - output_dir = '../outputs/figures'
#      - q_cols = [Q1–Q28], cat_cols = ['attendance','difficulty']
#      - plot_distribution(df, q_cols+cat_cols, output_dir)
#      - plot_correlations(df, q_cols+cat_cols, output_dir)
