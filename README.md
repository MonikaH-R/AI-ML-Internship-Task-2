# AI-ML-Internship-Task-2
 **Titanic Dataset: Exploratory Data Analysis (EDA)
**
This repository contains the solution for the Exploratory Data Analysis (EDA) task, focusing on the classic Titanic survival dataset. 
The primary goal is to understand the dataset's structure, distributions, and the relationships between passenger features and the target variable (Survived).

**Objectives**

1. Descriptive Statistics: Generate summaries (mean, median, standard deviation) for all features.

2. Distribution Analysis: Visualize numeric feature distributions using histograms and identify outliers using boxplots.

3. Feature Relationships: Analyze correlations between numeric features.

4. Survival Factors: Investigate key patterns and trends relating categorical features (Sex, Pclass) to survival outcomes.

** Tools and Libraries**

| Category | Tool/Library | Purpose | 
 | ----- | ----- | ----- | 
| **Language** | Python | Core programming language. | 
| **Data Handling** | **Pandas** | Data loading, manipulation, and summary statistics. | 
| **Visualization** | **Matplotlib** | Underlying plotting library. | 
| **Visualization** | **Seaborn** | High-level interface for creating informative statistical graphics. | 
| **System** | `os` | Handling file paths and creating the output directory. |


** Repository Structure
**
.
├── titanic.csv             # The original dataset used for analysis.
├── eda_titanic.py          # The complete Python script for all EDA steps.
└── /AI-ML-Internship-Task 2/  # The dedicated output folder (created by the script).
    ├── 01_numeric_histograms.png
    ├── 02_age_fare_boxplots.png
    ├── 03_correlation_matrix.png
    ├── 04_survival_by_sex.png
    └── 05_survival_by_pclass.png


** Key Findings**

The analysis confirmed several well-known survival patterns from the Titanic disaster:

1. Gender Bias: Females had a significantly higher survival rate than males.

2. Class Bias: Passengers in First Class (Pclass=1) were far more likely to survive than those in Second or Third Class.

3. Fare & Age: The Fare distribution is heavily right-skewed with significant outliers. The Age distribution is centered around 30 years.

4. Correlation: Pclass exhibited the strongest negative correlation with Survived, and Fare showed a moderate positive correlation, reinforcing the influence of socio-economic status on survival.

**How to Run the Script**

1. Prerequisites: Ensure Python is installed and you have the required libraries:
pip install pandas numpy matplotlib seaborn

2. Data: Place the titanic.csv file in the same directory as the eda_titanic.py script.
   
3. Execution: Run the script from your terminal or IDE:
python eda_titanic.py

4. Output: The script will sequentially display five plots and save them inside the dedicated output directory
