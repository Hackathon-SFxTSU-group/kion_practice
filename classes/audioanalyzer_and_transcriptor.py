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
    """
    Класс для анализа аудиодорожки видео с использованием моделей:
    - OpenL3 (извлечение эмбеддингов),
    - YAMNet (аудиоклассификация),
    - Whisper (распознавание речи),
    - RMS (энергия сигнала).
    """

    def __init__(self, video_path, output_dir="output_scenes", whisper_model_size="tiny",
                 ffmpeg_path="ffmpeg", ffprobe_path="ffprobe"):
        # Пути к файлам и параметрам
        self.video_path = video_path
        self.output_dir = output_dir
        self.audio_path = os.path.join(output_dir, "temp_audio.wav")
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path

        # Продолжительность видео
        self.video_duration = 0

        # Модели (загружаются позже)
        self.openl3_model = None
        self.yamnet_model = None

        # Данные для распознавания и логов
        self.segments = []           # Полные сегменты из Whisper
        self.segments_short = []     # Укороченные {start, end, text}
        self.detection_log = []      # Лог голосования для сцен

        # Устройство и модель Whisper
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper_model = whisper.load_model(whisper_model_size, device=self.device)

        # Создание директорий
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs("reports", exist_ok=True)

    def extract_audio(self):
        """
        Извлекает аудиодорожку из видео и сохраняет её в WAV-файл.
        """
        print("🎧 Извлечение аудио из видео...")
        self.audio_path = extract_audio_ffmpeg(self.video_path, self.audio_path, self.ffmpeg_path)
        self.video_duration = get_video_duration(self.video_path, self.ffprobe_path)
        print(f"✅ Аудио сохранено: {self.audio_path}")

    def load_models(self):
        """
        Загружает модели OpenL3 и YAMNet (если ещё не загружены).
        """
        print("📦 Загрузка моделей OpenL3 и YAMNet...")
        if self.openl3_model is None:
            self.openl3_model = openl3.models.load_audio_embedding_model("mel256", "music", 6144)
        if self.yamnet_model is None:
            self.yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
        print("✅ Модели успешно загружены.")

    def extract_openl3_features(self):
        """
        Извлекает аудиоэмбеддинги из OpenL3 по всей аудиодорожке.
        """
        print("🔍 Извлечение аудиопризнаков OpenL3...")
        y, sr = librosa.load(self.audio_path, sr=44100, mono=True)
        features, timestamps = openl3.get_audio_embedding(y, sr, model=self.openl3_model, hop_size=1.0)
        features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)  # нормализация
        print(f"✅ Извлечено {len(features)} векторов признаков.")
        return features, timestamps

    def extract_yamnet_labels(self):
        """
        Классифицирует аудиосигнал по временным отрезкам с помощью YAMNet.
        """
        print("🔊 Получение меток с YAMNet...")
        waveform, sr = librosa.load(self.audio_path, sr=16000)
        yamnet_output = self.yamnet_model(waveform)
        scores = yamnet_output[0].numpy()
        labels = np.argmax(scores, axis=1)
        print(f"✅ Получено {len(labels)} аудиометок.")
        return labels

    def compute_rms(self):
        """
        Вычисляет уровень RMS (энергии сигнала) по времени.
        """
        print("📊 Расчёт RMS-громкости...")
        y, sr = librosa.load(self.audio_path, sr=44100)
        rms = librosa.feature.rms(y=y)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
        print(f"✅ RMS рассчитано для {len(rms)} окон.")
        return times, rms

    def detect_scenes(self, features, timestamps, yamnet_labels, rms_t, rms_vals,
                      sensitivity=0.85, min_scene_duration=2.0):
        """
        Выявляет границы сцен с помощью голосования по трем критериям:
        - изменение OpenL3 признаков,
        - смена класса YAMNet,
        - скачок RMS-громкости.
        """
        print("✂️ Детектирование границ сцен...")
        smoothed = gaussian_filter1d(features, sigma=2, axis=0)
        diffs = [
            1 - cosine_similarity([smoothed[i - 1]], [smoothed[i]])[0][0]
            for i in tqdm(range(1, len(smoothed)), desc="🛑 Анализ")
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

        print(f"✅ Найдено сцен: {len(changes)}")
        return changes

    def process_asr(self, language="ru"):
        """
        Выполняет распознавание речи с помощью Whisper.
        """
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
        print(f"✅ Распознано сегментов: {len(self.segments)}")

    def detect_audio_activity(self, frame_duration=1.0):
        """
        Вычисляет RMS-энергию для фиксированных временных окон (кадров).
        """
        print(f"🎛 Расчёт аудиоэнергии по окнам по {frame_duration} сек...")
        y, sr = librosa.load(self.audio_path, sr=None)
        frame_len = int(sr * frame_duration)
        energy = [
            np.sqrt(np.mean(y[i:i + frame_len] ** 2))
            for i in range(0, len(y), frame_len)
        ]
        print(f"✅ Получено {len(energy)} окон.")
        return energy

    def export_full_report(self, changes, segments_short, sensitivity=0.85, min_scene_duration=2.0):
        """
        Сохраняет полный отчёт об аудиоанализе, включая сцены, речь и логи.
        """
        print("💾 Экспорт полного отчета...")
        scene_bounds = [0] + changes + [self.video_duration]
        scenes = [{
            "scene_id": i + 1,
            "start": round(scene_bounds[i], 3),
            "end": round(scene_bounds[i + 1], 3),
            "duration": round(scene_bounds[i + 1] - scene_bounds[i], 3)
        } for i in range(len(scene_bounds) - 1)]

        detection_log = [{
            "time": round(t, 3),
            "audio_change": bool(a),
            "yamnet_change": bool(y),
            "rms_change": bool(r),
            "voted": bool(v)
        } for t, a, y, r, v in self.detection_log]

        total_duration = sum(round(seg["end"] - seg["start"], 2) for seg in self.segments)
        total_words = sum(len(seg["text"].split()) for seg in self.segments)

        segment_reports = [{
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "duration": round(seg["end"] - seg["start"], 2),
            "word_count": len(seg["text"].split()),
            "text": seg["text"].strip()
        } for seg in self.segments]

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
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        print(f"✅ Отчёт сохранён: {path}")

    def run(self, sensitivity=0.85, min_scene_duration=2.0):
        """
        Запускает полный цикл аудиоанализа:
        - извлечение аудио,
        - загрузка моделей,
        - извлечение признаков,
        - сценовое деление,
        - распознавание речи,
        - экспорт отчета.
        """
        self.extract_audio()
        self.load_models()
        features, timestamps = self.extract_openl3_features()
        yamnet_labels = self.extract_yamnet_labels()
        rms_t, rms_vals = self.compute_rms()
        changes = self.detect_scenes(features, timestamps, yamnet_labels, rms_t, rms_vals,
                                     sensitivity, min_scene_duration)
        self.process_asr(language="ru")
        self.export_full_report(changes, self.segments_short, sensitivity, min_scene_duration)
        print(f"\n✅ Готово: {len(changes) + 1} сцен, {len(self.segments_short)} речевых сегментов.")
