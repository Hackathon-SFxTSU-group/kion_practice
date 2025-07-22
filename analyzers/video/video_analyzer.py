import cv2  # Для работы с видео
from tqdm import tqdm  # Для отображения прогресса
from scene_splitter.clip_scene_splitter import CLIPSceneSplitter  # Разделение сцен с помощью CLIP
from tracking.detector import ObjectDetector  # Детектор объектов (например, людей)
from tracking.tracker import DeepSortTracker  # Трекер для отслеживания объектов между кадрами


class VideoAnalyzer:
    def __init__(self):
        # Инициализация компонентов: разделитель сцен, детектор и трекер
        self.splitter = CLIPSceneSplitter()
        self.detector = ObjectDetector()
        self.tracker = DeepSortTracker()

    def analyze_video(self, video_path):
        # Разделяем видео на сцены с заданным шагом кадров
        scenes = self.splitter.detect_scenes(video_path)

        cap = cv2.VideoCapture(video_path)  # Открываем видео
        fps = cap.get(cv2.CAP_PROP_FPS)  # Получаем частоту кадров
        scene_data = []  # Список со сценами и id треков в них
        track_faces = {}  # Словарь для хранения вырезанных лиц по track_id
        tracking_frames = {}

        # Анализ каждой сцены по очереди
        for (start, end) in tqdm(scenes, desc="Analyzing scenes"):
            cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)  # Устанавливаем видео на старт сцены
            frame_idx = int(start * fps)  # Начальный индекс кадра
            end_idx = int(end * fps)  # Конечный индекс кадра
            track_ids_in_scene = set()  # Уникальные треки, замеченные в данной сцене

            while frame_idx < end_idx:
                ret, frame = cap.read()
                if not ret:
                    break  # Если видео закончилось раньше, выходим

                # Детекция объектов (людей) на кадре
                detections = self.detector.detect(frame)
                # Обновляем трекер — получаем список текущих треков
                tracks = self.tracker.update(detections, frame)

                for t in tracks:
                    track_id = t['track_id']
                    x1, y1, x2, y2 = map(int, t['bbox'])  # Получаем координаты

                    track_ids_in_scene.add(track_id)  # Добавляем track_id в текущую сцену

                    # Вырезаем лицо по координатам
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size == 0:
                        continue  # Пропускаем, если обрезка не удалась

                    if track_id not in track_faces:
                        track_faces[track_id] = []

                    track_faces[track_id].append(face_crop)  # Сохраняем лицо

                    # ⬇️ Сохраняем bbox по кадру
                    if track_id not in tracking_frames:
                        tracking_frames[track_id] = {}
                    tracking_frames[track_id][frame_idx] = (x1, y1, x2, y2)

                frame_idx += 1  # Переходим к следующему кадру

            # Сохраняем данные по сцене
            scene_data.append({
                "start": start,
                "end": end,
                "track_ids": list(track_ids_in_scene)
            })

        cap.release()  # Освобождаем видеофайл
        return scene_data, track_faces, tracking_frames  # Возвращаем список сцен и лица по трекам