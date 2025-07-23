from sentence_transformers import SentenceTransformer
import numpy as np

from utils import (
    enrich_scenes_with_characters,
    enrich_scenes_with_audio,
    group_semantic_scenes,
    cluster_scenes_with_time_windows,
    resolve_time_overlaps,
    clean_and_merge_short_scenes,
    print_scenes_formatted,
)


class SceneEnricher:
    def __init__(self, segments, energy, device="cuda"):
        self.segments = segments
        self.energy = energy
        self.device = device
        self.sentence_model = SentenceTransformer("all-mpnet-base-v2", device=device)

    def enrich(self, scene_data, track_id_to_person=None):
        # 1. Копируем сцены для обогащения
        scenes = [dict(s) for s in scene_data]

        # 2. Добавляем персонажей (если есть отображение track_id → имя)
        if track_id_to_person:
            scenes = enrich_scenes_with_characters(scenes, track_id_to_person)

        # 3. Добавляем текст и аудио-энергию
        scenes = enrich_scenes_with_audio(scenes, self.segments, self.energy)

        return scenes

    def group(self, enriched_scenes, method="semantic"):
        if method == "semantic":
            scenes_grouped = group_semantic_scenes(enriched_scenes, self.sentence_model)
        elif method == "window":
            scenes_grouped = cluster_scenes_with_time_windows(enriched_scenes, self.sentence_model)
        else:
            raise ValueError(f"Unknown grouping method: {method}")

        return scenes_grouped

    def postprocess(self, scenes_grouped):
        cleaned = resolve_time_overlaps(scenes_grouped)
        cleaned = clean_and_merge_short_scenes(cleaned)
        return cleaned

    def run(self, scene_data, track_id_to_person=None, print_report=True):
        # Объединённый вызов enrichment → grouping → postprocessing
        enriched = self.enrich(scene_data, track_id_to_person)
        grouped = self.group(enriched)
        cleaned = self.postprocess(grouped)

        if print_report:
            print_scenes_formatted(cleaned)

        return cleaned
