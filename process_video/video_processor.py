import cv2  # Для работы с видео
from tracking.detector import ObjectDetector  # Детектор объектов (например, людей)
from tracking.face_reid import FaceReId
from tracking.tracker import DeepSortTracker  # Трекер для отслеживания объектов между кадрами


class VideoProcessor:
    def __init__(self, using_cache=False):
        # Инициализация компонентов: разделитель сцен, детектор и трекер
        if not using_cache:
            self.detector = ObjectDetector()
            self.tracker = DeepSortTracker()
            self.face_reid = FaceReId()

    def generate_video_with_persons(self, video_path, output_path):
        track_faces, tracking_frames = self.get_faces_data(video_path)

        # Получаем отображение track_id → персонаж (имя или ID)
        track_id_to_person = self.face_reid.analyze_persons(track_faces)

        self.render_video_with_faces(video_path, output_path, tracking_frames, track_id_to_person)

        return track_faces, tracking_frames, track_id_to_person

    def get_faces_data(self, video_path):
        # Разделяем видео на сцены с заданным шагом кадров
        cap = cv2.VideoCapture(video_path)  # Открываем видео
        track_faces = {}  # Словарь для хранения вырезанных лиц по track_id
        tracking_frames = {}
        frame_idx = 0

        # Анализ каждой сцены по очереди
        while True:
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

        cap.release()  # Освобождаем видеофайл
        return track_faces, tracking_frames  # Возвращаем список сцен и лица по трекам

    def render_video_with_faces(self, video_path, output_path, tracking_frames, track_id_to_person):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # или 'avc1' если 'mp4v' не работает
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Отрисовка всех bbox, известных для этого кадра
            for track_id, frames in tracking_frames.items():
                if frame_idx in frames:
                    x1, y1, x2, y2 = frames[frame_idx]
                    name = track_id_to_person.get(track_id, f"person_{track_id}")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()