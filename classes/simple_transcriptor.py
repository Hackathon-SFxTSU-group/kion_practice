import whisper
import json
import os
from datetime import datetime


class ASRProcessor:
    def __init__(self, model_name="base"):
        self.model = whisper.load_model(model_name)
        self.segments = []
        self.segments_short = []
        self.audio_path = ""

    def process(self, audio_path: str, language="ru", verbose=True, word_timestamps=True):
        print(f"Распознавание речи для: {audio_path}")
        self.audio_path = audio_path
        result = self.model.transcribe(
            audio_path,
            language=language,
            verbose=verbose,
            word_timestamps=word_timestamps
        )
        self.segments = result["segments"]
        self.segments_short = self._get_short_segments(self.segments)

        # Сохраняем сегменты (при желании)
        segments_path = self._make_report_path(audio_path, suffix="segments.json")
        self._save_json(self.segments, segments_path)
        print(f" Сегменты сохранены в: {segments_path}")

        # Генерация отчета
        report_path = self._make_report_path(audio_path, suffix="asr_report.json")
        self.generate_report(report_path)

        return self.segments_short

    def generate_report(self, report_path):
        segment_reports = []
        total_duration = 0
        total_words = 0

        for seg in self.segments:
            start = round(seg["start"], 2)
            end = round(seg["end"], 2)
            duration = round(end - start, 2)
            text = seg["text"].strip()
            word_count = len(text.split())

            total_duration += duration
            total_words += word_count

            segment_reports.append({
                "start": start,
                "end": end,
                "duration": duration,
                "word_count": word_count,
                "text": text
            })

        overall = {
            "audio_path": self.audio_path,
            "total_segments": len(self.segments),
            "total_duration_sec": round(total_duration, 2),
            "avg_segment_duration_sec": round(total_duration / len(self.segments), 2) if self.segments else 0,
            "avg_word_count_per_segment": round(total_words / len(self.segments), 2) if self.segments else 0,
            "generated_at": datetime.now().isoformat()
        }

        report = {
            "summary": overall,
            "segments": segment_reports
        }

        self._save_json(report, report_path)
        print(f"Отчёт сохранён: {report_path}")

    def _make_report_path(self, audio_path, suffix):
        os.makedirs("reports", exist_ok=True)
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        return os.path.join("reports", f"{base_name}_{suffix}")

    @staticmethod
    def _save_json(data, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _get_short_segments(segments):
        return [
            {
                "start": round(seg["start"], 2),
                "end": round(seg["end"], 2),
                "text": seg["text"]
            }
            for seg in segments
        ]
