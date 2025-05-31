import os

from huggingface_hub import HfApi, HfFolder, create_repo, upload_folder

# Configuration
LOCAL_MODELS_DIR = "nunchaku-models"
HUGGINGFACE_NAMESPACE = "mit-han-lab"
PRIVATE = False  # Set to True if you want the repos to be private

# Initialize API
api = HfApi()

# Get your token from local cache
token = HfFolder.get_token()

# Iterate over all folders in the models directory
for model_name in os.listdir(LOCAL_MODELS_DIR):
    model_path = os.path.join(LOCAL_MODELS_DIR, model_name)
    if not os.path.isdir(model_path):
        continue  # Skip non-folder files

    repo_id = f"{HUGGINGFACE_NAMESPACE}/{model_name}"
    print(f"\n📦 Uploading {model_path} to {repo_id}")

    # Create the repo (skip if it exists)
    try:
        create_repo(repo_id, token=token, repo_type="model", private=PRIVATE, exist_ok=True)
    except Exception as e:
        print(f"⚠️ Failed to create repo {repo_id}: {e}")
        continue

    # Upload the local model folder
    try:
        upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            token=token,
            repo_type="model",
            path_in_repo="",  # root of repo
        )
        print(f"✅ Uploaded {model_name} successfully.")
    except Exception as e:
        print(f"❌ Upload failed for {model_name}: {e}")
