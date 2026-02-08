
from huggingface_hub import HfApi

# Define your space ID (e.g., 'username/wellness-predictor-app')
SPACE_REPO_ID = "Tamilvelan/tourism-project"

api = HfApi()

# Create the Space if it doesn't exist
api.create_repo(repo_id=SPACE_REPO_ID, repo_type="space", space_sdk="docker", exist_ok=True)

# Upload all deployment files
files_to_upload = ["app.py", "requirements.txt", "Dockerfile"]
for file in files_to_upload:
    api.upload_file(
        path_or_fileobj=file,
        path_in_repo=file,
        repo_id=SPACE_REPO_ID,
        repo_type="space",
        token=HF_TOKEN
    )

print(f"Deployment complete! View your app at: https://huggingface.co/spaces/{SPACE_REPO_ID}")
