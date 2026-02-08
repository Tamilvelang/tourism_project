
# --- INSTALL NECESSARY PACKAGES ---
!pip install huggingface_hub datasets

!pip install mlflow

import os
from huggingface_hub import HfApi, login


# --- 2. PREPARE YOUR DATA ---
# NOTE: Upload 'tourism.csv' to the Colab files sidebar first.
data_folder = "/content/tourism_project_predict/data"

import shutil
if os.path.exists('/content/tourism_project_predict/data/tourism.csv'):
    shutil.move('/content/tourism_project_predict/data/tourism.csv', os.path.join(data_folder, "tourism.csv"))
else:
    # Creating a dummy path check if you haven't uploaded it yet
    print("Please upload 'tourism.csv' to the Colab file area on the left.")

# --- 3. REGISTER DATA ON HUGGING FACE ---
# Hugging Face Username and Dataset Name
HF_TOKEN = "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"
DATASET_REPO_ID = "Tamilvelan/tourism-wellness-data"

login(token=HF_TOKEN)
api = HfApi()

# Uploading the raw data to Hugging Face
api.upload_file(
    path_or_fileobj=os.path.join(data_folder, "tourism.csv"),
    path_in_repo="tourism.csv",
    repo_id=DATASET_REPO_ID,
    repo_type="dataset"
)

print(f"Success! Data registered at: https://huggingface.co/datasets/{DATASET_REPO_ID}")
