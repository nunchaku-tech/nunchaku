import argparse
import os

from huggingface_hub import snapshot_download
from modelscope.hub.api import HubApi
from modelscope.hub.constants import Licenses, ModelVisibility


def sync_model(repo_name: str, hf_repo: str, ms_repo: str):
    print(f"\n🔄 Syncing {hf_repo} -> {ms_repo}")

    # Login to ModelScope
    MODELSCOPE_TOKEN = os.getenv("MODELSCOPE_TOKEN")
    assert MODELSCOPE_TOKEN, "Please set the MODELSCOPE_TOKEN environment variable or hardcode the token."

    api = HubApi()
    api.login(MODELSCOPE_TOKEN)

    # Download the model snapshot from Hugging Face
    local_dir = snapshot_download(repo_id=hf_repo, cache_dir=repo_name, local_dir=repo_name)
    print(f"📥 Downloaded to: {local_dir}")

    # Check if the ModelScope repo already exists
    exists = False
    try:
        api.get_model(ms_repo)
        exists = True
        print(f"✅ Model already exists on ModelScope: {ms_repo}")
    except Exception:
        print(f"ℹ️ Model not found on ModelScope: {ms_repo}, creating...")

    # Create repo if it doesn't exist
    if not exists:
        api.create_model(
            model_id=ms_repo,
            visibility=ModelVisibility.PUBLIC,  # Change to "Private" if needed
            license=Licenses.APACHE_V2,
        )
        print(f"✅ Created ModelScope repo: {ms_repo}")

    # Upload model files to ModelScope
    print(f"⏫ Uploading to ModelScope...")
    # api.upload_model(model_dir=local_dir, model_id=ms_repo)
    api.upload_folder(
        repo_id=ms_repo,
        folder_path=local_dir,
        commit_message=f"Sync from Hugging Face {hf_repo}",
        ignore_patterns=["*nunchaku-tech*"],
    )
    print(f"✅ Sync complete: {hf_repo} -> {ms_repo}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--repo-name",
        type=str,
        required=True,
        help="Name of the HuggingFace repository under nunchaku-tech to sync to (e.g., `nunchaku`)",
    )
    args = parser.parse_args()
    sync_model(args.repo_name, f"nunchaku-tech/{args.repo_name}", f"nunchaku-tech/{args.repo_name}")
