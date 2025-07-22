from ultralytics import YOLO  # Импорт модели YOLO (Ultralytics YOLOv8)
import torch  # Для определения доступности устройства (CUDA/CPU)


class ObjectDetector:
    def __init__(self, model_name="yolov8n.pt", device="cuda" if torch.cuda.is_available() else "cpu"):
        # Определяем устройство: GPU (если доступно) или CPU
        self.device = device

        # Загружаем модель YOLO (по умолчанию: yolov8n — самая лёгкая)
        self.model = YOLO(model_name)

        # Перемещаем модель на нужное устройство
        self.model.to(self.device)

    def detect(self, frame):
        # Выполняем детекцию объектов на входном кадре
        # verbose=False отключает вывод в консоль
        results = self.model(frame, verbose=False)[0]

        detections = []  # Список детекций (объекты, прошедшие фильтр)

        for box in results.boxes:
            # Получаем ID класса (например, 0 = человек, 1 = велосипед и т.д.)
            cls_id = int(box.cls.item())

            # Уверенность (confidence) модели в этой детекции
            conf = float(box.conf.item())

            # Пропускаем все классы, кроме "человек" (cls_id == 0)
            if cls_id != 0:
                continue

            # Получаем координаты ограничивающего прямоугольника
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w = x2 - x1  # ширина
            h = y2 - y1  # высота

            # Добавляем результат в формате: [bbox, confidence, class_id]
            detections.append([[x1, y1, w, h], conf, cls_id])

        return detections  # Возвращаем список всех подходящих объектов
