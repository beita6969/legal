import os
from huggingface_hub import snapshot_download

# Set proxy
os.environ["http_proxy"] = "http://127.0.0.1:10808"
os.environ["https_proxy"] = "http://127.0.0.1:10808"

repo_id = "qwedsacf/competition_math"
local_dir = "11/integrated_aflow_roll/data/raw/competition_math"

print(f"Downloading {repo_id} to {local_dir}...")
snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=local_dir, local_dir_use_symlinks=False)
print("Download complete!")


