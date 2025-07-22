from deep_sort_realtime.deepsort_tracker import DeepSort  # Импорт трекера Deep SORT
import torch  # Для определения доступности GPU


class DeepSortTracker:
    def __init__(self):
        # Инициализация трекера Deep SORT с заданными параметрами
        self.tracker = DeepSort(
            max_age=30,  # Сколько кадров "запоминать" исчезнувший объект перед удалением
            n_init=3,  # Сколько раз объект должен появиться подряд, чтобы считаться подтверждённым
            nms_max_overlap=1.0,  # Максимально допустимое перекрытие при NMS (не используется в этой реализации)
            max_cosine_distance=0.2,  # Порог на косинусную дистанцию между эмбеддингами
            nn_budget=None,  # Ограничение на размер памяти для сравнения эмбеддингов (None — без ограничения)
            override_track_class=None,  # Не используется — можно оставить None
            embedder="torchreid",  # Тип модели эмбеддера (mobilenet — быстрый, torchreid — точнее, но медленнее)
            half=True,  # Использовать половинную точность (fp16) — быстрее на GPU
            bgr=True,  # Кадры передаются в формате BGR (как в OpenCV)
            embedder_gpu=torch.cuda.is_available()  # Включение GPU для эмбеддера, если доступен
        )

    def update(self, detections, frame):
        """
        Обновление трекера на текущем кадре.

        detections: список детекций в формате [[x, y, w, h, conf], ...]
        frame: изображение (numpy-массив), соответствующее кадру

        Возвращает список словарей:
        [
            {
                'track_id': уникальный ID объекта,
                'bbox': [x1, y1, x2, y2] — координаты ограничивающего прямоугольника
            },
            ...
        ]
        """
        # Обновляем трекер новыми детекциями и кадром
        tracks = self.tracker.update_tracks(detections, frame=frame)

        output = []

        for track in tracks:
            if not track.is_confirmed():
                continue  # Пропускаем неподтверждённые треки

            # Получаем координаты в формате (x1, y1, x2, y2)
            bbox = track.to_ltrb()

            # Добавляем результат в выходной список
            output.append({
                'track_id': track.track_id,
                'bbox': bbox,
            })

        return output  # Список подтверждённых треков с координатами и ID