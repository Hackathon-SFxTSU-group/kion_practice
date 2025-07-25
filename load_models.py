import os
import torch
import whisper
import openl3
import tensorflow_hub as hub
import tensorflow as tf
import shutil
import tarfile
import urllib.request
from sentence_transformers import SentenceTransformer
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
from insightface.app import FaceAnalysis


def prepare_dirs():
    os.makedirs("models/whisper/small", exist_ok=True)
    os.makedirs("models/openl3", exist_ok=True)
    os.makedirs("models/yamnet", exist_ok=True)
    os.makedirs("models/sentence-transformers/all-mpnet-base-v2", exist_ok=True)
    os.makedirs("models/facenet", exist_ok=True)
    os.makedirs("models/clip", exist_ok=True)
    os.makedirs("models/yolo", exist_ok=True)
    os.makedirs("models/insightface", exist_ok=True)
    os.makedirs("models/torchreid", exist_ok=True)


def download_whisper_small():
    print(f"🔊 Загрузка Whisper-small...")
    os.environ["WHISPER_CACHE"] = "models/whisper/small"
    _ = whisper.load_model("small")
    print("✅ Whisper-small загружен и сохранён в models/whisper/small\n")


def download_openl3_model():
    print("🎶 Загрузка OpenL3 модели...")
    os.environ["KERAS_HOME"] = "models/openl3"
    model = openl3.models.load_audio_embedding_model(
        input_repr="mel256", content_type="music", embedding_size=6144
    )
    model.save("models/openl3/openl3_model.h5")
    print("✅ OpenL3 сохранён в models/openl3/\n")


def download_yamnet_model():
    print("🔊 Загрузка YAMNet модели...")
    url = "https://tfhub.dev/google/yamnet/1?tf-hub-format=compressed"
    local_tar = "models/yamnet/yamnet.tar.gz"

    urllib.request.urlretrieve(url, local_tar)
    with tarfile.open(local_tar, "r:gz") as tar:
        tar.extractall(path="models/yamnet")

    os.remove(local_tar)
    print("✅ YAMNet скачан и распакован в models/yamnet/\n")


def download_sentence_transformer():
    print("🧠 Загрузка модели SentenceTransformer: all-mpnet-base-v2...")
    model_path = "models/sentence-transformers/all-mpnet-base-v2"
    _ = SentenceTransformer("all-mpnet-base-v2", cache_folder=model_path)
    print(f"✅ SentenceTransformer загружен в {model_path}\n")

def download_yolo_model():
    print("👁 Загрузка YOLOv8n модели...")
    model = YOLO("yolov8n.pt")
    model.save("models/yolo/yolov8n.pt")
    print("✅ YOLOv8n загружен и сохранён в models/yolo/\n")


def download_clip_model():
    print("🎬 Загрузка CLIP (ViT-B/32)...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.save_pretrained("models/clip")
    processor.save_pretrained("models/clip")
    print("✅ CLIP (ViT-B/32) сохранён в models/clip/\n")


def download_insightface_model():
    print("🧠 Загрузка InsightFace модели (buffalo_l)...")
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0 if torch.cuda.is_available() else -1)
    print("✅ InsightFace 'buffalo_l' успешно загружена (в ~/.insightface)\n")


def download_deepsort_torchreid():
    print("📦 Загрузка DeepSort torchreid...")
    from deep_sort_realtime.embedder.embedder_pytorch import TorchReIDEmbedder
    _ = TorchReIDEmbedder(model_name='osnet_x1_0', device='cuda' if torch.cuda.is_available() else 'cpu')
    print("✅ TorchReID embedder загружен\n")


if __name__ == "__main__":
    prepare_dirs()
    download_whisper_small()
    download_openl3_model()
    download_yamnet_model()
    download_sentence_transformer()
    download_yolo_model()
    download_clip_model()
    download_insightface_model()
    download_deepsort_torchreid()
    print("🎉 Все модели успешно загружены и готовы к использованию локально!")
