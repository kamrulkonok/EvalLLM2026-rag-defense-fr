from huggingface_hub import login, create_repo, upload_file # type: ignore
import os

REPO_ID = "kamrulkonok/evalllm2026-french-defense"
FILE_PATH = "data/pages.jsonl"

login() 

create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)

upload_file(
    path_or_fileobj=FILE_PATH,
    path_in_repo="pages.jsonl",
    repo_id=REPO_ID,
    repo_type="dataset",
)

print(f"Uploaded {FILE_PATH} to {REPO_ID}")