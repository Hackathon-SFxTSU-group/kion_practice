import os
import cv2
import json
import time
from tqdm import tqdm

from scene_splitter.clip_scene_splitter import CLIPSceneSplitter
from tracking.detector import ObjectDetector
from tracking.tracker import DeepSortTracker
# from tracking.face_reid_pytorch import FaceReId  # Реидентификация лиц (опционально)

class VideoPipeline:
    """
    Класс для сценового анализа видео: выделение логических сцен, трекинг объектов (например, лиц),
    сбор вырезанных лиц и координат треков. Подходит для генерации JSON-отчетов и последующей аналитики.
    """

    def __init__(self, using_cache=False, render_video=False):
        """
        Инициализация компонентов видеопайплайна.

        :param using_cache: если True, не создаёт новые экземпляры компонентов
        :param render_video: если True, включается финальный рендер размеченного видео
        """
        self.render_video = render_video

        # Компоненты пайплайна, создаются, если не используем кэш
        if not using_cache:
            self.splitter = CLIPSceneSplitter()       # Выделение логических сцен на основе CLIP
            self.detector = ObjectDetector()          # Детектор объектов (например, лиц)
            self.tracker = DeepSortTracker()          # DeepSORT-трекер объектов
            # self.face_reid = FaceReId()              # Реидентификация лиц (если нужно)

        # Служебные переменные для хранения результатов
        self.scene_data = []          # Список сцен с трек-данными
        self.track_faces = {}         # Вырезанные лица по track_id
        self.tracking_frames = {}     # Словарь: track_id -> {frame_idx: (x1, y1, x2, y2)}

    def run(self, video_path, output_dir="output_files"):
        """
        Запускает полный цикл анализа видео: сценовое разбиение, трекинг, экспорт отчета.

        :param video_path: Путь к исходному видеофайлу
        :param output_dir: Каталог для хранения выходных файлов
        :return: Кортеж: (scene_data, track_faces, tracking_frames)
        """
        os.makedirs(output_dir, exist_ok=True)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_video_path = os.path.join(output_dir, f"{video_name}_annotated.mp4")

        # === Шаг 1: Сценовое разбиение ===
        print("🎬 Детектирование сцен...")
        t0 = time.time()
        scenes = self.splitter.detect_scenes(video_path)  # [(start, end), ...]
        scene_split_time = time.time() - t0
        print(f"✅ Найдено {len(scenes)} сцен за {scene_split_time:.2f} сек")

        # === Шаг 2: Трекинг по сценам ===
        print("🧠 Анализ сцен и трекинг объектов...")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        t1 = time.time()
        for (start, end) in tqdm(scenes, desc="Analyzing scenes"):
            cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)  # Устанавливаем позицию в миллисекундах
            frame_idx = int(start * fps)
            end_idx = int(end * fps)
            track_ids_in_scene = set()

            while frame_idx < end_idx:
                ret, frame = cap.read()
                if not ret:
                    break

                # Детекция и трекинг объектов
                detections = self.detector.detect(frame)
                tracks = self.tracker.update(detections, frame)

                for t in tracks:
                    track_id = t['track_id']
                    x1, y1, x2, y2 = map(int, t['bbox'])
                    track_ids_in_scene.add(track_id)

                    # Вырезаем лицо
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size == 0:
                        continue

                    # Сохраняем лицо по track_id
                    self.track_faces.setdefault(track_id, []).append(face_crop)

                    # Сохраняем координаты по кадрам
                    self.tracking_frames.setdefault(track_id, {})[frame_idx] = (x1, y1, x2, y2)

                frame_idx += 1

            # Сохраняем сцену и все найденные треки в ней
            self.scene_data.append({
                "start": start,
                "end": end,
                "track_ids": list(track_ids_in_scene)
            })

        tracking_time = time.time() - t1
        cap.release()

        # === Шаг 3: Сохранение JSON-отчета ===
        print("💾 Сохранение JSON отчета...")
        self.export_report_to_json(video_path, output_dir, scene_split_time, tracking_time)

        # === Шаг 4: (опционально) Рендер аннотированного видео ===
        # track_id_to_person = {track_id: f"person_{track_id}" for track_id in self.track_faces}
        # if self.render_video:
        #     print("🎥 Рендер видео с аннотациями...")
        #     self.render_labeled_video(video_path, output_video_path, self.tracking_frames, track_id_to_person)

        print("✅ Пайплайн успешно завершен.")
        return self.scene_data, self.track_faces, self.tracking_frames

    def export_report_to_json(self, video_path, output_dir, scene_split_time, tracking_time):
        """
        Формирует и сохраняет JSON-отчет с информацией о сценах и треках.

        :param video_path: Путь к видео
        :param output_dir: Папка для сохранения
        :param scene_split_time: Время на сценовый сплиттер
        :param tracking_time: Время на трекинг
        """
        report = {
            "video_path": video_path,
            "scene_data": self.scene_data,
            "track_faces": {str(k): len(v) for k, v in self.track_faces.items()},
            "tracking_frames": {str(k): list(v.keys()) for k, v in self.tracking_frames.items()},
            "timing": {
                "scene_split_time_sec": round(scene_split_time, 3),
                "tracking_time_sec": round(tracking_time, 3)
            }
        }

        os.makedirs("reports", exist_ok=True)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join("reports", f"{video_name}_video_report.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, ensure_ascii=False)

        print(f"📄 Отчёт сохранён: {output_path}")

    def render_labeled_video(self, video_path, output_path, tracking_frames, track_id_to_person):
        """
        Генерирует видео с отрисованными прямоугольниками и подписями по track_id.

        :param video_path: Путь к исходному видео
        :param output_path: Куда сохранить размеченное видео
        :param tracking_frames: Треки с координатами bbox по кадрам
        :param track_id_to_person: Соответствие треков и имён
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

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
        print(f"📼 Размеченное видео сохранено: {output_path}")
