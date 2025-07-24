import os
import json
import torch
import whisper
import librosa
import openl3
import numpy as np
import tensorflow_hub as hub
from tqdm import tqdm
from datetime import datetime
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics.pairwise import cosine_similarity
from utils.audio_tools import extract_audio_ffmpeg, get_video_duration


class AudioAnalyzer:
    def __init__(self, video_path, output_dir="output_scenes", whisper_model_size="tiny",
                 ffmpeg_path="ffmpeg", ffprobe_path="ffprobe"):
        self.video_path = video_path
        self.output_dir = output_dir
        self.audio_path = os.path.join(output_dir, "temp_audio.wav")
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
        self.video_duration = 0
        self.openl3_model = None
        self.yamnet_model = None
        self.detection_log = []

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper_model = whisper.load_model(whisper_model_size, device=self.device)

        self.segments = []
        self.segments_short = []

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs("reports", exist_ok=True)

    def extract_audio(self):
        print("🎧 Извлечение аудио...")
        self.audio_path = extract_audio_ffmpeg(self.video_path, self.audio_path, self.ffmpeg_path)
        self.video_duration = get_video_duration(self.video_path, self.ffprobe_path)
        print(f"✅ Аудио сохранено: {self.audio_path}")

    def load_models(self):
        print("📦 Загрузка моделей...")
        if self.openl3_model is None:
            self.openl3_model = openl3.models.load_audio_embedding_model("mel256", "music", 6144)
        if self.yamnet_model is None:
            self.yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
        print("✅ Модели загружены")

    def extract_openl3_features(self):
        print("🔍 OpenL3: извлечение признаков...")
        y, sr = librosa.load(self.audio_path, sr=44100, mono=True)
        features, timestamps = openl3.get_audio_embedding(y, sr, model=self.openl3_model, hop_size=1.0)
        features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
        print(f"✅ Получено {len(features)} признаков")
        return features, timestamps

    def extract_yamnet_labels(self):
        print("🔊 YAMNet: извлечение классов...")
        waveform, sr = librosa.load(self.audio_path, sr=16000)
        yamnet_output = self.yamnet_model(waveform)
        scores = yamnet_output[0].numpy()
        labels = np.argmax(scores, axis=1)
        print(f"✅ Получено {len(labels)} меток")
        return labels

    def compute_rms(self):
        print("📈 RMS: расчёт...")
        y, sr = librosa.load(self.audio_path, sr=44100)
        rms = librosa.feature.rms(y=y)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
        print(f"✅ RMS рассчитано: {len(rms)} значений")
        return times, rms

    def detect_scenes(self, features, timestamps, yamnet_labels, rms_t, rms_vals,
                      sensitivity=0.85, min_scene_duration=2.0):
        print("✂️ Детектирование сцен...")
        smoothed = gaussian_filter1d(features, sigma=2, axis=0)
        diffs = [
            1 - cosine_similarity([smoothed[i - 1]], [smoothed[i]])[0][0]
            for i in tqdm(range(1, len(smoothed)), desc="🚧 Анализ")
        ]
        threshold = np.percentile(diffs, sensitivity * 100)

        changes = []
        prev_time = -min_scene_duration
        self.detection_log = []

        for i, diff in enumerate(diffs):
            t = timestamps[i + 1]
            audio_change = diff > threshold
            yamnet_change = (yamnet_labels[i] != yamnet_labels[i + 1])
            rms_change = np.abs(rms_vals[min(i, len(rms_vals) - 1)] - rms_vals[max(i - 1, 0)]) > 0.1
            vote = sum([audio_change, yamnet_change, rms_change]) >= 2
            self.detection_log.append((t, audio_change, yamnet_change, rms_change, vote))
            if vote and (t - prev_time >= min_scene_duration):
                changes.append(t)
                prev_time = t

        print(f"✅ Найдено {len(changes)} границ сцен")
        return changes

    def process_asr(self, language="ru"):
        print("📝 Распознавание речи (Whisper)...")
        result = self.whisper_model.transcribe(
            self.audio_path,
            language=language,
            verbose=True,
            word_timestamps=True
        )
        self.segments = result["segments"]
        self.segments_short = [
            {"start": round(seg["start"], 2), "end": round(seg["end"], 2), "text": seg["text"]}
            for seg in self.segments
        ]
        print(f"✅ Распознано {len(self.segments)} сегментов речи")

    def detect_audio_activity(self, frame_duration=1.0):
        print(f"🎚 Расчёт энергии по кадрам длительностью {frame_duration} сек...")
        y, sr = librosa.load(self.audio_path, sr=None)
        frame_length = int(sr * frame_duration)
        energy = [
            np.sqrt(np.mean(y[i:i + frame_length] ** 2))
            for i in range(0, len(y), frame_length)
        ]
        print(f"✅ Расчёт завершён. Кадров: {len(energy)}")
        return energy

    def export_full_report(self, changes, segments_short, sensitivity=0.85, min_scene_duration=2.0):
        print("💾 Экспорт полного JSON-отчета...")

        scene_bounds = [0] + changes + [self.video_duration]
        scenes = []
        for i in range(len(scene_bounds) - 1):
            start, end = scene_bounds[i], scene_bounds[i + 1]
            scenes.append({
                "scene_id": i + 1,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(end - start, 3)
            })

        detection_log = []
        for t, a, y, r, v in self.detection_log:
            detection_log.append({
                "time": round(t, 3),
                "audio_change": bool(a),
                "yamnet_change": bool(y),
                "rms_change": bool(r),
                "voted": bool(v)
            })

        total_duration = 0
        total_words = 0
        segment_reports = []

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

        asr_summary = {
            "total_segments": len(self.segments),
            "total_duration_sec": round(total_duration, 2),
            "avg_segment_duration_sec": round(total_duration / len(self.segments), 2) if self.segments else 0,
            "avg_word_count_per_segment": round(total_words / len(self.segments), 2) if self.segments else 0
        }

        report_data = {
            "video_path": self.video_path,
            "method": "audio_openl3_yamnet_rms_whisper",
            "generated_at": datetime.now().isoformat(),
            "sensitivity": sensitivity,
            "min_scene_duration": min_scene_duration,
            "scene_count": len(scenes),
            "asr_summary": asr_summary,
            "scenes": scenes,
            "speech_segments": segments_short,
            "speech_segments_detailed": segment_reports,
            "detection_log": detection_log,
            "model_used": ["openl3", "yamnet", "rms", "whisper"]
        }

        filename = os.path.splitext(os.path.basename(self.video_path))[0]
        path = os.path.join("reports", f"{filename}_full_report.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        print(f"✅ Полный отчёт сохранён: {path}")

    def run(self, sensitivity=0.85, min_scene_duration=2.0):
        self.extract_audio()
        self.load_models()
        features, timestamps = self.extract_openl3_features()
        yamnet_labels = self.extract_yamnet_labels()
        rms_t, rms_vals = self.compute_rms()
        changes = self.detect_scenes(features, timestamps, yamnet_labels, rms_t, rms_vals,
                                     sensitivity, min_scene_duration)
        self.process_asr(language="ru")
        self.export_full_report(changes, self.segments_short, sensitivity, min_scene_duration)
        print(f"\n✅ Всего сцен: {len(changes) + 1}, распознанных фрагментов: {len(self.segments_short)}")
