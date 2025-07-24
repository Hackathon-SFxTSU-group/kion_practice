from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np

from utils.scene_tools import (
    enrich_scenes_with_characters,
    enrich_scenes_with_audio,
    group_semantic_scenes,
    cluster_scenes_with_time_windows,
    resolve_time_overlaps,
    clean_and_merge_short_scenes,
    save_scenes_report_to_json
)


class SceneEnricher:
    def __init__(self, segments, energy, device="cuda"):
        self.segments = segments
        self.energy = energy
        self.device = device

        print("🧠 Загрузка sentence-transformer модели...")
        self.sentence_model = SentenceTransformer("all-mpnet-base-v2", device=device)
        print("✅ SentenceTransformer готов.")

    def enrich(self, scene_data, track_id_to_person=None):
        print("🎭 Обогащение сцен...")

        # 1. Копируем сцены для обогащения
        scenes = [dict(s) for s in scene_data]

        # 2. Добавляем персонажей (если есть отображение track_id → имя)
        if track_id_to_person:
            print("👤 Добавление персонажей в сцены...")
            scenes = enrich_scenes_with_characters(scenes, track_id_to_person)

        # 3. Добавляем текст и аудио-энергию
        print("🔊 Добавление речевых сегментов и аудиоэнергии...")
        scenes = enrich_scenes_with_audio(scenes, self.segments, self.energy)

        print(f"✅ Обогащено сцен: {len(scenes)}")
        return scenes

    def group(self, enriched_scenes, method="semantic"):
        print(f"🧩 Группировка сцен методом: {method}")

        if method == "semantic":
            scenes_grouped = group_semantic_scenes(enriched_scenes, self.sentence_model)
        elif method == "window":
            scenes_grouped = cluster_scenes_with_time_windows(enriched_scenes, self.sentence_model)
        else:
            raise ValueError(f"Unknown grouping method: {method}")

        print(f"✅ Сцен сгруппировано: {len(scenes_grouped)}")
        return scenes_grouped

    def postprocess(self, scenes_grouped):
        print("🧼 Постобработка: разрешение пересечений и объединение коротких сцен...")

        cleaned = resolve_time_overlaps(scenes_grouped)
        cleaned = clean_and_merge_short_scenes(cleaned)

        print(f"✅ Финальное количество сцен после очистки: {len(cleaned)}")
        return cleaned

    def run(self, scene_data, track_id_to_person=None, print_report=True):
        print("🚀 Запуск полного процесса обогащения и группировки...")
        
        enriched = self.enrich(scene_data, track_id_to_person)
        grouped = self.group(enriched)
        cleaned = self.postprocess(grouped)

        if print_report:
            print("💾 Сохранение отчета по сценам...")
            save_scenes_report_to_json(cleaned)
            print("📁 JSON-отчет успешно сохранен.")

        print("✅ Обработка сцен завершена.")
        return cleaned
