import torch  # Для определения доступности CUDA (GPU)
from insightface.app import FaceAnalysis  # Модуль для распознавания лиц и извлечения эмбеддингов
from tqdm import tqdm  # Прогресс-бар при обработке лиц
import numpy as np  # Для работы с массивами и векторами
from sklearn.cluster import DBSCAN  # Кластеризация для объединения треков по похожим лицам


class FaceReId:
    def __init__(self):
        # Инициализация FaceAnalysis из insightface с нужным провайдером (GPU или CPU)
        self.face_app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"]
        )

        # Подготовка модели: ctx_id=0 для GPU, -1 для CPU
        self.face_app.prepare(ctx_id=0 if torch.cuda.is_available() else -1)

    def analyze_persons(self, track_faces):
        # Получаем усреднённые эмбеддинги лиц по каждому track_id (одному человеку)
        track_embeddings = self.get_avg_embeddings(track_faces)

        # Кластеризуем полученные эмбеддинги, чтобы сопоставить одинаковых людей
        return self.cluster_track_ids(track_embeddings)

    def get_avg_embeddings(self, track_faces):
        # Словарь: {track_id: средний эмбеддинг}
        track_embeddings = {}

        # Проходим по каждому треку (объекту/человеку)
        for track_id, crops in tqdm(track_faces.items(), desc="Analyzing faces"):
            embeddings = []

            # Для каждого кадра с лицом в треке извлекаем эмбеддинг
            for face in crops:
                faces = self.face_app.get(face)
                if faces:
                    embeddings.append(faces[0].embedding)  # Берём эмбеддинг первого найденного лица

            # Если есть хотя бы один эмбеддинг — усредняем и сохраняем
            if embeddings:
                avg_emb = np.mean(embeddings, axis=0)
                track_embeddings[track_id] = avg_emb

        return track_embeddings  # Возвращаем: {track_id: avg_embedding}

    def cluster_track_ids(self, track_embeddings):
        # Получаем эмбеддинги и соответствующие им track_id
        X = list(track_embeddings.values())  # Список эмбеддингов
        ids = list(track_embeddings.keys())  # Список треков

        # Кластеризация с помощью DBSCAN по косинусной метрике
        clustering = DBSCAN(eps=0.7, min_samples=1, metric='cosine').fit(X)

        # Сопоставление: track_id → имя человека (например, person_0, person_1 и т.д.)
        track_id_to_person = {
            track_id: f"person_{label}"
            for track_id, label in zip(ids, clustering.labels_)
        }

        return track_id_to_person  # Возвращаем словарь сопоставлений
