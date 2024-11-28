import os
from huggingface_hub import hf_hub_download


def download_model_files():
    """Downloads the necessary model files from Hugging Face Hub.

    Returns paths to the downloaded model and tokenizer files.
    """
    model_dir = "downloaded_model"
    repo_id = "byvictorrr/html-login-field-detector"

    # Ensure the directory exists
    os.makedirs(model_dir, exist_ok=True)

    # Remove existing files to force overwrite
    files_to_download = ["model.safetensors", "config.json", "tokenizer.json"]
    for filename in files_to_download:
        file_path = os.path.join(model_dir, filename)
        if os.path.exists(file_path):
            os.remove(file_path)

    # Download fresh files
    hf_hub_download(repo_id=repo_id, filename="model.safetensors", cache_dir=model_dir)
    hf_hub_download(repo_id=repo_id, filename="config.json", cache_dir=model_dir)
    hf_hub_download(repo_id=repo_id, filename="tokenizer.json", cache_dir=model_dir)

    return model_dir
