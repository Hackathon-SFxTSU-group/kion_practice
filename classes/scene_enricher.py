from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np

# Импорт вспомогательных функций для обработки сцен
from utils.scene_tools import (
    enrich_scenes_with_characters,        # Добавление персонажей в сцены
    enrich_scenes_with_audio,             # Привязка аудиосегментов и энергии
    group_semantic_scenes,                # Группировка по смыслу (семантически)
    cluster_scenes_with_time_windows,     # Группировка по временным окнам
    resolve_time_overlaps,                # Устранение перекрывающихся сцен
    clean_and_merge_short_scenes,         # Объединение/удаление слишком коротких сцен
    save_scenes_report_to_json            # Сохранение итогового отчета в JSON
)

class SceneEnricher:
    """
    Класс SceneEnricher обогащает сцены (выделенные ранее из видео) 
    дополнительной информацией: персонажами, речью, аудиоэнергией,
    а также выполняет группировку и постобработку.
    """

    def __init__(self, segments, energy, device="cuda"):
        """
        Инициализация класса.

        :param segments: Список распознанных речевых сегментов (из ASR)
        :param energy: Массив аудиоэнергии по времени
        :param device: Устройство для sentence-transformers (обычно 'cuda' или 'cpu')
        """
        self.segments = segments
        self.energy = energy
        self.device = device

        print("🧠 Загрузка sentence-transformer модели...")
        self.sentence_model = SentenceTransformer("all-mpnet-base-v2", device=device)
        print("✅ SentenceTransformer готов.")

    def enrich(self, scene_data, track_id_to_person=None):
        """
        Обогащение исходных сцен: добавление персонажей, речевых сегментов и энергии.

        :param scene_data: Список сцен (каждая с 'start' и 'end')
        :param track_id_to_person: Словарь отображения ID треков в имена персонажей (необязательно)
        :return: Обогащенные сцены
        """
        print("🎭 Обогащение сцен...")

        # Создаем копии сцен, чтобы не изменять оригинал
        scenes = [dict(s) for s in scene_data]

        # Если доступна информация о персонажах, добавляем её в сцены
        if track_id_to_person:
            print("👤 Добавление персонажей в сцены...")
            scenes = enrich_scenes_with_characters(scenes, track_id_to_person)

        # Добавляем текст из распознанной речи и аудиоэнергию
        print("🔊 Добавление речевых сегментов и аудиоэнергии...")
        scenes = enrich_scenes_with_audio(scenes, self.segments, self.energy)

        print(f"✅ Обогащено сцен: {len(scenes)}")
        return scenes

    def group(self, enriched_scenes, method="semantic"):
        """
        Группировка обогащённых сцен по смыслу или времени.

        :param enriched_scenes: Список обогащенных сцен
        :param method: Метод группировки: "semantic" или "window"
        :return: Сгруппированные сцены
        """
        print(f"🧩 Группировка сцен методом: {method}")

        if method == "semantic":
            # Группировка по смыслу текста (embedding + кластеризация)
            scenes_grouped = group_semantic_scenes(enriched_scenes, self.sentence_model)
        elif method == "window":
            # Группировка по временным окнам с учетом похожести
            scenes_grouped = cluster_scenes_with_time_windows(enriched_scenes, self.sentence_model)
        else:
            raise ValueError(f"Unknown grouping method: {method}")

        print(f"✅ Сцен сгруппировано: {len(scenes_grouped)}")
        return scenes_grouped

    def postprocess(self, scenes_grouped):
        """
        Финальная обработка сцен: устранение перекрытий, очистка, объединение коротких.

        :param scenes_grouped: Список сцен после группировки
        :return: Очищенный список сцен
        """
        print("🧼 Постобработка: разрешение пересечений и объединение коротких сцен...")

        # Удаляем пересекающиеся по времени сцены
        cleaned = resolve_time_overlaps(scenes_grouped)

        # Объединяем слишком короткие сцены с соседними
        cleaned = clean_and_merge_short_scenes(cleaned)

        print(f"✅ Финальное количество сцен после очистки: {len(cleaned)}")
        return cleaned

    def run(self, scene_data, track_id_to_person=None, print_report=True):
        """
        Полный процесс обогащения и обработки сцен.

        :param scene_data: Список исходных сцен
        :param track_id_to_person: Словарь треков → имена персонажей (необязательно)
        :param print_report: Если True, сохраняется JSON-отчет
        :return: Финальный список сцен
        """
        print("🚀 Запуск полного процесса обогащения и группировки...")

        # Шаг 1: Добавление персонажей, текста и энергии
        enriched = self.enrich(scene_data, track_id_to_person)

        # Шаг 2: Группировка сцен по смыслу или времени
        grouped = self.group(enriched)

        # Шаг 3: Финальная очистка и приведение к единому виду
        cleaned = self.postprocess(grouped)

        # Шаг 4: Сохранение результата в JSON-файл
        if print_report:
            print("💾 Сохранение отчета по сценам...")
            save_scenes_report_to_json(cleaned)
            print("📁 JSON-отчет успешно сохранен.")

        print("✅ Обработка сцен завершена.")
        return cleaned
