# data-science-turkey-student-evaluation

## Project Overview
This project analyzes the Turkiye Student Evaluation dataset to understand student feedback on courses and instructors, build thematic features, and develop predictive models for course difficulty. It includes data preprocessing, exploratory analysis, feature engineering, modeling, and result visualization.

## Folder Structure
```
project-root/
├── data/
│   └── TurkiyeStudentEvaluation.csv      # raw evaluation data
├── scripts/                             # analysis pipelines
│   ├── 01_data_preprocessing.py         # load & clean raw data
│   ├── 02_exploratory_analysis.py       # distributions & correlation heatmap
│   ├── 03_feature_engineering.py        # thematic scores, scaling, train/test split
│   ├── 04_modeling.py                   # train classifier & clustering models
│   └── 05_evaluation_visualization.py   # performance plots & report generation
├── outputs/                             # generated files
│   ├── processed/                       # cleaned CSV
│   ├── figures/                         # distribution, heatmap, ROC, clusters
│   ├── features/                        # train/test CSV splits
│   ├── models/                          # saved model artifacts & report JSON
│   └── report.md                        # markdown summary of model performance
├── requirements.txt                     # project dependencies
└── README.md                            # this documentation
```

## Usage
1. **Setup the Project:**
   - Clone the repository.
   - Ensure you have Python installed.
   - Install required dependencies using the requirements.txt file.
   ```bash
   pip install -r requirements.txt
   ```
2. **Run data preprocessing:**
   ```bash
   python scripts/01_data_preprocessing.py
   ```
3. **Run exploratory analysis:**
   ```bash
   python scripts/02_exploratory_analysis.py
   ```
4. **Run feature engineering:**
   ```bash
   python scripts/03_feature_engineering.py
   ```
5. **Run modeling:**
   ```bash
   python scripts/04_modeling.py
   ```
6. **Run evaluation & visualization:**
   ```bash
   python scripts/05_evaluation_visualization.py
   ```

