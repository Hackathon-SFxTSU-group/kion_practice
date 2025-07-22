import numpy as np  # Для работы с массивами и матрицами
import torch  # Для использования CUDA (GPU) и модели CLIP
import cv2  # OpenCV для обработки видео и извлечения кадров
from PIL import Image  # Для преобразования кадров в формат, подходящий для CLIP
import clip  # Библиотека с моделью CLIP от OpenAI
from tqdm import tqdm  # Прогресс-бар при обработке кадров
from sklearn.cluster import KMeans  # Алгоритм кластеризации для разбивки сцен


class CLIPSceneSplitter:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        # Определяем устройство: CUDA (GPU) или CPU
        self.device = device

        # Загружаем модель CLIP и функцию предварительной обработки изображений
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def extract_frames(self, video_path):
        # Извлекаем кадры из видео через равные промежутки (каждые every_n_frames кадров)
        cap = cv2.VideoCapture(video_path)
        frames = []  # Список кадров
        timestamps = []  # Список временных меток соответствующих кадров
        frame_idx = 0
        fps = cap.get(cv2.CAP_PROP_FPS)  # Частота кадров видео

        while True:
            ret, frame = cap.read()  # Читаем кадр
            if not ret:
                break  # Выход из цикла, если видео закончилось
            # Преобразуем BGR (OpenCV) в RGB (PIL/CLIP)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
            timestamps.append(frame_idx / fps)  # Сохраняем время кадра
            frame_idx += 1

        cap.release()
        return frames, timestamps  # Возвращаем список кадров и соответствующих времён

    def compute_embeddings(self, frames):
        # Получаем векторные представления (эмбеддинги) для каждого кадра
        embeddings = []
        with torch.no_grad():  # Отключаем градиенты для ускорения
            for frame in tqdm(frames, desc="Computing embeddings"):
                # Преобразуем кадр с помощью preprocess, добавляем batch размерность и переносим на устройство
                image = self.preprocess(Image.fromarray(frame)).unsqueeze(0).to(self.device)
                embedding = self.model.encode_image(image)  # Получаем эмбеддинг с помощью CLIP
                embeddings.append(embedding.cpu().numpy()[0])  # Переносим на CPU и сохраняем
        return np.stack(embeddings)  # Объединяем в один массив

    def cluster_embeddings(self, embeddings, n_clusters=8):
        # Разбиваем эмбеддинги на кластеры (сцены) с помощью KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)  # Получаем метки кластеров для каждого кадра
        return cluster_labels

    def get_scenes(self, cluster_labels, timestamps):
        # Формируем список сцен на основе смены кластеров
        scenes = []
        start = 0  # Индекс начала текущей сцены
        for i in range(1, len(cluster_labels)):
            if cluster_labels[i] != cluster_labels[i - 1]:
                # Если метка кластера изменилась — новая сцена
                scenes.append((timestamps[start], timestamps[i]))
                start = i  # Начинаем новую сцену
        # Добавляем последнюю сцену до конца видео
        scenes.append((timestamps[start], timestamps[-1]))
        return scenes  # Возвращаем список сцен как (start_time, end_time)

    def detect_scenes(self, video_path, n_clusters=8):
        # Основной метод: извлекает сцены из видео
        frames, timestamps = self.extract_frames(video_path)
        embeddings = self.compute_embeddings(frames)
        labels = self.cluster_embeddings(embeddings, n_clusters)
        scenes = self.get_scenes(labels, timestamps)
        return scenes  # Возвращает список временных промежутков сцен