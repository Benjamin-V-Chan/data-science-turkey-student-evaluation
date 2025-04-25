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

