import os
import gdown
models = {
    "baseline_lr0.001_bs8_freezeTrue": "1tEjaqZeYisvmQ9DGd4RcBh354CPiTlzB",
    "pretrained_freeze_lr0.001_bs8_freezeTrue": "1U5zVoaGNvEZWzo7ylDRBNm3_zPbGSKvS",
    "pretrained_unfreeze_lr0.001_bs8_freezeTrue": "1tEjaqZeYisvmQ9DGd4RcBh354CPiTlzB",
    "pretrained_freeze_lr0.0005_bs16_freezeTrue": "15XObSz6UrqnPqOQPF-YMWKFlWUGOTvde",
}

def create_model_dirs(model_name):
    os.makedirs(f"models/{model_name}", exist_ok=True)

def download_model(file_id, output):
    if not os.path.exists(output):
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading model to {output}...")
        gdown.download(url, output, quiet=False)
    else:
        print(f"File already exists at: {output}")

def download_all_models():
    for model_name, file_id in models.items():
        create_model_dirs(model_name)
        output_path = f"models/{model_name}/food101_checkpoint_best.pth"
        download_model(file_id, output_path)

