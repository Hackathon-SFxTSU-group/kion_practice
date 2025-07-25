import os
import cv2
import json
import time
from tqdm import tqdm

from scene_splitter.clip_scene_splitter import CLIPSceneSplitter  # Сценовый сплиттер на основе CLIP
from tracking.detector import ObjectDetector                       # Обнаружение объектов (например, лиц)
from tracking.tracker import DeepSortTracker                      # Трекинг объектов по видео
from tracking.face_reid import FaceReId                           # Реидентификация лиц по embedding'ам


class VideoPipeline:
    """
    Класс VideoPipeline реализует пайплайн:
    - разделение видео на сцены,
    - детектирование и трекинг объектов,
    - сохранение лиц по track_id,
    - построение отчета и (опционально) рендер размеченного видео.
    """

    def __init__(self, using_cache=False, render_video=False):
        """
        Инициализация компонентов пайплайна.

        :param using_cache: если True — компоненты не загружаются повторно
        :param render_video: если True — создаётся размеченное видео с ID персонажей
        """
        self.render_video = render_video

        if not using_cache:
            self.splitter = CLIPSceneSplitter()    # Разделение видео на логические сцены
            self.detector = ObjectDetector()       # Обнаружение объектов (лиц)
            self.tracker = DeepSortTracker()       # Отслеживание объектов
            self.face_reid = FaceReId()            # Реидентификация лиц

        # Хранилища результатов анализа
        self.scene_data = []          # Данные по сценам и track_id
        self.track_faces = {}         # Лица по track_id (для анализа)
        self.tracking_frames = {}     # Привязка треков к кадрам

    def run(self, video_path, output_dir="output_files"):
        """
        Запуск полного пайплайна: сценовое разделение, трекинг, экспорт отчета и рендер.

        :param video_path: Путь к видеофайлу
        :param output_dir: Каталог для вывода результатов
        :return: Кортеж с результатами: scene_data, track_faces, tracking_frames, track_id_to_person
        """
        os.makedirs(output_dir, exist_ok=True)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_video_path = os.path.join(output_dir, f"{video_name}_annotated.mp4")

        # === Шаг 1: Разделение на сцены ===
        print("🎬 Детектирование сцен...")
        t0 = time.time()
        scenes = self.splitter.detect_scenes(video_path)
        scene_split_time = time.time() - t0
        print(f"✅ Найдено сцен: {len(scenes)} за {scene_split_time:.2f} сек")

        # Подготовка видео
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        # === Шаг 2: Анализ каждой сцены ===
        print("🧠 Анализ сцен и трекинг объектов...")
        t1 = time.time()
        for (start, end) in tqdm(scenes, desc="Analyzing scenes"):
            cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
            frame_idx = int(start * fps)
            end_idx = int(end * fps)
            track_ids_in_scene = set()

            while frame_idx < end_idx:
                ret, frame = cap.read()
                if not ret:
                    break

                # Детекция и трекинг объектов на кадре
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

                    # Сохраняем bbox в нужный кадр
                    self.tracking_frames.setdefault(track_id, {})[frame_idx] = (x1, y1, x2, y2)

                frame_idx += 1

            # Добавляем сцену с track_id персонажей
            self.scene_data.append({
                "start": start,
                "end": end,
                "track_ids": list(track_ids_in_scene)
            })

        tracking_time = time.time() - t1
        cap.release()

        # === Шаг 3: Экспорт отчёта ===
        print("💾 Сохранение отчета...")
        self.export_report_to_json(video_path, output_dir, scene_split_time, tracking_time)

        # === Шаг 4: Реидентификация лиц ===
        print("🧬 Реидентификация лиц...")
        track_id_to_person = self.face_reid.analyze_persons(self.track_faces)

        # === Шаг 5: Рендер видео (если нужно) ===
        if self.render_video:
            print("🎥 Генерация размеченного видео...")
            self.render_labeled_video(video_path, output_video_path, self.tracking_frames, track_id_to_person)

        print("✅ Пайплайн завершён.")
        return self.scene_data, self.track_faces, self.tracking_frames, track_id_to_person

    def export_report_to_json(self, video_path, output_dir, scene_split_time, tracking_time):
        """
        Сохраняет полный JSON-отчёт по результатам анализа видео.

        :param video_path: Исходный путь к видео
        :param output_dir: Каталог вывода
        :param scene_split_time: Время, затраченное на сценовое разделение
        :param tracking_time: Время, затраченное на трекинг
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
        Отрисовывает видео с прямоугольниками вокруг лиц и подписями идентифицированных персонажей.

        :param video_path: Путь к исходному видео
        :param output_path: Путь для сохранения размеченного видео
        :param tracking_frames: Координаты треков по кадрам
        :param track_id_to_person: Отображение track_id → имя персонажа
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

            # Рисуем прямоугольники и имена
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
