
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from huggingface_hub import HfApi

# --- 1. LOAD DATASET FROM HUGGING FACE ---
# DATASET_REPO_ID and HF_TOKEN are defined from the previous step
try:
    dataset = load_dataset(DATASET_REPO_ID)
    df = pd.DataFrame(dataset['train'])
    print("Dataset loaded successfully from Hugging Face.")
except Exception as e:
    print(f"Error loading dataset: {e}")

# --- 2. DATA CLEANING & PREPROCESSING ---
# Removing 'CustomerID' as its primary key and may not be required (unnecessary column)
if 'CustomerID' in df.columns:
    df.drop(columns=['CustomerID'], inplace=True)

# Handling Missing Values
# Numerical: Fill with Median (Robust to outliers)
num_cols = ['Age', 'MonthlyIncome', 'DurationOfPitch']
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Categorical/Discrete: Fill with Mode
cat_cols = ['NumberOfTrips', 'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfChildrenVisiting']
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# --- 3. SPLIT DATASET ---
# 80% Train, 20% Test. Stratify ensures the 'ProdTaken' ratio is preserved.
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['ProdTaken']
)

# Save processed files locally in the "tourism_project_predict/data" folder
train_path = "/content/tourism_project_predict/data/train.csv"
test_path = "/content/tourism_project_predict/data/test.csv"

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)
print("Data split and saved locally.")

# --- 4. UPLOAD PROCESSED DATA BACK TO HUGGING FACE ---
api = HfApi()
for file_name, local_path in [("train.csv", train_path), ("test.csv", test_path)]:
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=file_name,
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN
    )

print("Cleaned Train and Test datasets uploaded to Hugging Face space.")
