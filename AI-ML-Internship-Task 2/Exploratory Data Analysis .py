import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
# Set the target folder path as requested.
# Using a raw string (r"...") is correct for Windows paths.
OUTPUT_DIR = r"C:\Users\HP\PycharmProjects\pythonProject\AI & ML INTERNSHIP\AI-ML-Internship-Task 2"
DATA_FILE = 'titanic.csv'
FILE_NAMES = [
    '01_numeric_histograms.png',
    '02_age_fare_boxplots.png',
    '03_correlation_matrix.png',
    '04_survival_by_sex.png',
    '05_survival_by_pclass.png'
]

# --- 1. Setup Environment ---
try:
    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory ensured: {OUTPUT_DIR}")

except Exception as e:
    print(f"CRITICAL ERROR: Could not create or access directory: {e}")
    OUTPUT_DIR = os.getcwd()
    print(f"Falling back to current working directory: {OUTPUT_DIR}. Please check the path and permissions.")


# --- 2. Load the Dataset ---
try:
    # NOTE: Assuming the data file is accessible from the current context.
    df = pd.read_csv(DATA_FILE)
    print(f"\nDataset '{DATA_FILE}' loaded successfully.")
except FileNotFoundError:
    print(f"\nError: '{DATA_FILE}' not found. Please ensure the data file is in the same folder as this script.")
    exit()

# --- 3. Initial Inspection and Summary Statistics (Not changing these) ---
print("\n--- DataFrame Head ---")
print(df.head())
print("\n--- DataFrame Info ---")
df.info()

print("\n--- Summary Statistics (All Columns) ---")
print(df.describe(include='all'))


# --- 4. Visualization of Numeric Features ---
print("\nStarting Plot Generation (Files will be saved BEFORE being shown)...")


# --- 4.1 Histograms (Plot 1) ---
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
fig1.suptitle('Plot 1: Histograms of Numeric Features (Distribution Check)', fontsize=16)

# Plotting using the axes objects
sns.histplot(df['Age'].dropna(), kde=True, ax=axes1[0, 0], bins=30)
axes1[0, 0].set_title('Age Distribution')

sns.histplot(df['Fare'], kde=True, ax=axes1[0, 1], bins=30)
axes1[0, 1].set_title('Fare Distribution (Right Skew)')

sns.histplot(df['SibSp'], kde=False, ax=axes1[1, 0], discrete=True, shrink=0.8)
axes1[1, 0].set_title('SibSp Distribution')
axes1[1, 0].set_xticks(range(df['SibSp'].min(), df['SibSp'].max() + 1))

sns.histplot(df['Parch'], kde=False, ax=axes1[1, 1], discrete=True, shrink=0.8)
axes1[1, 1].set_title('Parch Distribution')
axes1[1, 1].set_xticks(range(df['Parch'].min(), df['Parch'].max() + 1))

fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
# CRITICAL FIX: Save using the explicit figure object and save BEFORE show
fig1.savefig(os.path.join(OUTPUT_DIR, FILE_NAMES[0]))
print(f"Saved '{FILE_NAMES[0]}' to the specified output directory.")
plt.show()
plt.close(fig1)


# --- 4.2 Boxplots (Plot 2) ---
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 6))
fig2.suptitle('Plot 2: Boxplots for Age and Fare (Outlier Check)', fontsize=16)

sns.boxplot(y=df['Age'].dropna(), ax=axes2[0], color='skyblue')
axes2[0].set_title('Age Boxplot')

sns.boxplot(y=df['Fare'], ax=axes2[1], color='lightcoral')
axes2[1].set_title('Fare Boxplot (High Outliers Present)')

fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
# CRITICAL FIX: Save using the explicit figure object and save BEFORE show
fig2.savefig(os.path.join(OUTPUT_DIR, FILE_NAMES[1]))
print(f"Saved '{FILE_NAMES[1]}' to the specified output directory.")
plt.show()
plt.close(fig2)


# --- 4.3 Correlation Matrix (Plot 3) ---
fig3, ax3 = plt.subplots(figsize=(8, 6))
numeric_features = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
correlation_matrix = df[numeric_features].corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, linecolor='black', ax=ax3)
ax3.set_title('Plot 3: Correlation Matrix of Numeric Features')

fig3.tight_layout()
# CRITICAL FIX: Save using the explicit figure object and save BEFORE show
fig3.savefig(os.path.join(OUTPUT_DIR, FILE_NAMES[2]))
print(f"Saved '{FILE_NAMES[2]}' to the specified output directory.")
plt.show()
plt.close(fig3)


# --- 5. Visualization of Categorical Features vs. Survival ---
# --- 5.1 Survival Rate by Sex (Plot 4) ---
fig4, ax4 = plt.subplots(figsize=(7, 6))
sns.countplot(x='Sex', hue='Survived', data=df, palette='viridis', ax=ax4)
ax4.set_title('Plot 4: Survival Count by Sex (Clear Gender Bias)')
ax4.set_xlabel('Sex')
ax4.legend(title='Survived', labels=['Died', 'Survived'])

fig4.tight_layout()
# CRITICAL FIX: Save using the explicit figure object and save BEFORE show
fig4.savefig(os.path.join(OUTPUT_DIR, FILE_NAMES[3]))
print(f"Saved '{FILE_NAMES[3]}' to the specified output directory.")
plt.show()
plt.close(fig4)

# --- 5.2 Survival Rate by Passenger Class (Plot 5) ---
fig5, ax5 = plt.subplots(figsize=(7, 6))
sns.countplot(x='Pclass', hue='Survived', data=df, palette='magma', ax=ax5)
ax5.set_title('Plot 5: Survival Count by Passenger Class')
ax5.set_xlabel('Passenger Class (1=1st, 2=2nd, 3=3rd)')
ax5.legend(title='Survived', labels=['Died', 'Survived'])

fig5.tight_layout()
# CRITICAL FIX: Save using the explicit figure object and save BEFORE show
fig5.savefig(os.path.join(OUTPUT_DIR, FILE_NAMES[4]))
print(f"Saved '{FILE_NAMES[4]}' to the specified output directory.")
plt.show()
plt.close(fig5)


# --- 6. Final Verification ---
print("\n--- Verification ---")
all_files_present = True
for filename in FILE_NAMES:
    full_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(full_path):
        print(f"✅ FOUND: {filename}")
    else:
        print(f"❌ MISSING: {filename}")
        all_files_present = False

if all_files_present:
    print("\nSUCCESS: All 5 output files were successfully generated and saved to the folder.")
else:
    print("\nWARNING: One or more files are missing. Please ensure Python has write permissions for the directory.")
