import os
import json
from glob import glob


class UnifiedSceneMerger:
    """
    Класс UnifiedSceneMerger объединяет сценические таймкоды из разных источников:
    - аудиосегментация,
    - видеосегментация,
    - речевые сегменты (ASR),
    - тематические блоки (из LLM или др.).

    Результатом является единый финальный файл с перечнем сцен и метаинформацией.
    """

    def __init__(self, video_filename: str, reports_dir="reports", output_path=None, merge_gap=1.5):
        """
        Инициализация класса.

        :param video_filename: Путь к видеофайлу (нужен только для имени)
        :param reports_dir: Директория с промежуточными JSON-отчетами
        :param output_path: Путь для финального JSON-файла (если не задан — формируется автоматически)
        :param merge_gap: Максимально допустимый разрыв между границами для объединения (в секундах)
        """
        self.video_filename = os.path.splitext(os.path.basename(video_filename))[0]
        self.reports_dir = reports_dir
        self.output_path = output_path or os.path.join(reports_dir, f"{self.video_filename}_final_scene_decision.json")
        self.merge_gap = merge_gap

        # Источники данных
        self.audio_scenes = []
        self.video_scenes = []
        self.asr_segments = []
        self.themes = []

    def load_reports(self):
        """
        Загружает все JSON-файлы из папки reports и классифицирует по типу данных:
        - аудио/видео сцены
        - темы
        - распознанные сегменты речи
        """
        for path in glob(os.path.join(self.reports_dir, "*.json")):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

                if "method" in data and "scenes" in data:
                    # Отчёт от аудиоанализатора
                    self.audio_scenes = data["scenes"]
                elif "scene_data" in data:
                    # Отчёт от видеотрекера
                    self.video_scenes = data["scene_data"]
                elif "segments" in data:
                    # Результат распознавания речи
                    self.asr_segments = data["segments"]
                elif "themes" in data:
                    # Тематическая группировка
                    self.themes = data["themes"]

    def collect_boundaries(self):
        """
        Сбор всех временных границ начала и конца сцен из разных источников.
        Возвращает отсортированный список уникальных границ (в секундах).
        """
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
        """
        Объединение границ, которые слишком близко расположены друг к другу.
        Вместо двух близких границ будет взята их средняя.

        :param boundaries: Список отсортированных временных меток
        :return: Объединённый список границ
        """
        merged = []
        for b in boundaries:
            if not merged or abs(b - merged[-1]) > self.merge_gap:
                merged.append(b)
            else:
                merged[-1] = round((merged[-1] + b) / 2, 2)  # Среднее значение между двумя близкими точками
        return merged

    def build_scenes(self, merged_boundaries):
        """
        Формирует список финальных сцен на основе объединённых границ и источников.

        :param merged_boundaries: Границы сцен
        :return: Список словарей со сценами
        """
        scenes = []
        for i in range(len(merged_boundaries) - 1):
            start = round(merged_boundaries[i], 2)
            end = round(merged_boundaries[i + 1], 2)
            sources = []

            # Определяем, какие источники участвуют в данной сцене
            if self.contains_scene(self.audio_scenes, start, end):
                sources.append("audio")
            if self.contains_scene(self.video_scenes, start, end):
                sources.append("video")
            if self.contains_scene(self.themes, start, end):
                sources.append("themes")
            if self.contains_scene(self.asr_segments, start, end):
                sources.append("speech")

            # Находим описание темы, если она попадает в сцену
            description = next(
                (t["theme"] for t in self.themes if self.in_range(start, end, t["time"])),
                None
            )

            scenes.append({
                "start": start,
                "end": end,
                "sources": sources,
                "description": description
            })
        return scenes

    @staticmethod
    def contains_scene(scenes, start, end):
        """
        Проверка, содержится ли сцена в списке сцен из источника.

        :param scenes: Список сцен
        :param start: Начало сцены
        :param end: Конец сцены
        :return: True, если сцена с такими границами есть в источнике
        """
        for s in scenes:
            if abs(s["start"] - start) < 1e-1 and abs(s["end"] - end) < 1e-1:
                return True
        return False

    @staticmethod
    def in_range(start, end, time_range):
        """
        Проверка, попадает ли указанный временной интервал в заданный диапазон времени.

        :param start: Начало интервала
        :param end: Конец интервала
        :param time_range: Диапазон [start, end]
        :return: True, если границы совпадают с допустимой погрешностью
        """
        return abs(time_range[0] - start) < 1e-1 and abs(time_range[1] - end) < 1e-1

    def export(self, scenes):
        """
        Сохраняет финальный результат в JSON-файл.

        :param scenes: Список финальных сцен
        """
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(scenes, f, indent=2, ensure_ascii=False)
        print(f"✅ Финальный файл сцен сохранён: {self.output_path}")

    def run(self):
        """
        Запускает весь процесс объединения:
        - загружает отчёты,
        - собирает все границы,
        - объединяет близкие,
        - формирует сцены,
        - сохраняет результат.
        :return: Список финальных сцен
        """
        self.load_reports()
        raw_bounds = self.collect_boundaries()
        merged_bounds = self.merge_close_boundaries(raw_bounds)
        final_scenes = self.build_scenes(merged_bounds)
        self.export(final_scenes)
        return final_scenes
