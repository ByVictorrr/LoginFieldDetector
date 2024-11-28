from huggingface_hub import hf_hub_download


def download_model_files():
    """Downloads the necessary model files from Hugging Face Hub.

    Returns paths to the downloaded model and tokenizer files.
    """
    model_dir = "downloaded_model"
    repo_id = "byvictorrr/html-login-field-detector"
    model_file = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin",
                                 cache_dir=model_dir)
    config_file = hf_hub_download(repo_id=repo_id, filename="config.json", cache_dir=model_dir)
    tokenizer_file = hf_hub_download(repo_id=repo_id, filename="tokenizer.json",
                                     cache_dir=model_dir)
    return model_dir
