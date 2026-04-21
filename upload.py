from huggingface_hub import HfApi
import os

api = HfApi()

token = os.environ.get("HF_TOKEN")

repo_id = os.environ.get("HF_REPO_ID", "your-org/BackpropTrace")

folder_to_upload = os.environ.get("HF_UPLOAD_PATH", ".")

if not token:
    raise RuntimeError("Set HF_TOKEN before running upload.py")

print(f"🚀 Starting upload to {repo_id}...")

api.upload_folder(
    folder_path=folder_to_upload,
    repo_id=repo_id,
    token=token
)

print(f"Upload complete: https://huggingface.co/{repo_id}")