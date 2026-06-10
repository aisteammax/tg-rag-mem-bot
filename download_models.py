import os
from huggingface_hub import snapshot_download

def get_reranker_model():
    if os.path.exists(".env"):
        with open(".env", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("RERANKER_MODEL="):
                    return line.strip().split("=")[1].strip()
    return os.getenv("RERANKER_MODEL", "DiTy/cross-encoder-russian-msmarco")

def download_models():
    model_name = get_reranker_model()
    local_dir = "local_model"
    
    if os.path.exists(local_dir):
        print("Model already exists in local_model. Skipping download.")
        return

    print(f"Downloading reranker model '{model_name}' to local directory...")
    try:
        snapshot_download(repo_id=model_name, local_dir=local_dir)
        print("Model downloaded successfully.")
    except Exception as e:
        print(f"Error downloading model: {e}")
        import sys
        sys.exit(1)

if __name__ == "__main__":
    download_models()
