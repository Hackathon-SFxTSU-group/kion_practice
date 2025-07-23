import os
import json
from glob import glob


class UnifiedSceneMerger:
    def __init__(self, video_filename: str, reports_dir="reports", output_path=None, merge_gap=1.5):
        self.video_filename = os.path.splitext(os.path.basename(video_filename))[0]
        self.reports_dir = reports_dir
        self.output_path = output_path or os.path.join(reports_dir, f"{self.video_filename}_final_scene_decision.json")
        self.merge_gap = merge_gap  # макс. разрыв между сценами для объединения

        self.audio_scenes = []
        self.video_scenes = []
        self.asr_segments = []
        self.themes = []

    def load_reports(self):
        # Загружаем все отчеты
        for path in glob(os.path.join(self.reports_dir, "*.json")):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                if "method" in data and "scenes" in data:
                    self.audio_scenes = data["scenes"]
                elif "scene_data" in data:
                    self.video_scenes = data["scene_data"]
                elif "segments" in data:
                    self.asr_segments = data["segments"]
                elif "themes" in data:
                    self.themes = data["themes"]

    def collect_boundaries(self):
        boundaries = set()

        def add_bounds(scenes):
            for s in scenes:
                boundaries.add(round(s["start"], 2))
                boundaries.add(round(s["end"], 2))

        add_bounds(self.audio_scenes)
        add_bounds(self.video_scenes)
        add_bounds(self.themes)
        add_bounds(self.asr_segments)

        return sorted(boundaries)

    def merge_close_boundaries(self, boundaries):
        merged = []
        for b in boundaries:
            if not merged or abs(b - merged[-1]) > self.merge_gap:
                merged.append(b)
            else:
                merged[-1] = (merged[-1] + b) / 2  # среднее значение
        return merged

    def build_scenes(self, merged_boundaries):
        scenes = []
        for i in range(len(merged_boundaries) - 1):
            start = round(merged_boundaries[i], 2)
            end = round(merged_boundaries[i + 1], 2)
            sources = []

            if self.contains_scene(self.audio_scenes, start, end):
                sources.append("audio")
            if self.contains_scene(self.video_scenes, start, end):
                sources.append("video")
            if self.contains_scene(self.themes, start, end):
                sources.append("themes")
            if self.contains_scene(self.asr_segments, start, end):
                sources.append("speech")

            # берем описание темы, если попадает в эту сцену
            description = next((t["theme"] for t in self.themes if self.in_range(start, end, t["time"])), None)

            scenes.append({
                "start": start,
                "end": end,
                "sources": sources,
                "description": description
            })
        return scenes

    @staticmethod
    def contains_scene(scenes, start, end):
        for s in scenes:
            if abs(s["start"] - start) < 1e-1 and abs(s["end"] - end) < 1e-1:
                return True
        return False

    @staticmethod
    def in_range(start, end, time_range):
        return abs(time_range[0] - start) < 1e-1 and abs(time_range[1] - end) < 1e-1

    def export(self, scenes):
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(scenes, f, indent=2, ensure_ascii=False)
        print(f"✅ Финальный файл сцен сохранён: {self.output_path}")

    def run(self):
        self.load_reports()
        raw_bounds = self.collect_boundaries()
        merged_bounds = self.merge_close_boundaries(raw_bounds)
        final_scenes = self.build_scenes(merged_bounds)
        self.export(final_scenes)
        return final_scenes
