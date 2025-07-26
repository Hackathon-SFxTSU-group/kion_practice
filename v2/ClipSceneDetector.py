import os
import cv2
import torch
import clip
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity


class ClipSceneDetector:
    def __init__(self, video_path, model_name="ViT-B/32", device=None,
                 frame_interval_sec=1.0, threshold=0.3):
        """
        :param video_path: Путь к видеофайлу.
        :param model_name: CLIP модель (ViT-B/32 или другая).
        :param device: "cuda", "cpu" или None.
        :param frame_interval_sec: Интервал кадров в секундах.
        :param threshold: Порог для детектирования смыслового перехода.
        """
        self.video_path = video_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)

        self.frame_interval_sec = frame_interval_sec
        self.threshold = threshold

    def extract_clip_embeddings(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps

        embeddings = []
        timestamps = []

        # Берём кадры с заданным шагом
        current_sec = 0
        while current_sec < duration_sec:
            frame_idx = int(current_sec * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_input = self.preprocess(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                emb = self.model.encode_image(img_input)
                emb = emb.cpu().numpy()
                emb = emb / np.linalg.norm(emb)
                embeddings.append(emb)
                timestamps.append(current_sec)

            current_sec += self.frame_interval_sec

        cap.release()
        return embeddings, timestamps

    def detect_scenes(self):
        embeddings, timestamps = self.extract_clip_embeddings()

        diffs = []
        for i in range(1, len(embeddings)):
            sim = cosine_similarity(embeddings[i - 1], embeddings[i])[0][0]
            diff = 1 - sim
            diffs.append(diff)

        cuts = []
        for i, d in enumerate(diffs):
            if d > self.threshold:
                cuts.append(timestamps[i + 1])

        return cuts, diffs, timestamps

    def run(self):
        cuts, diffs, timestamps = self.detect_scenes()
        print(f"✅ Найдено {len(cuts)} смысловых переходов.")
        return cuts, diffs, timestamps
