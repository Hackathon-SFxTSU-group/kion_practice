import os
import json
import torch
import whisper
import librosa
import torchaudio
import torchopenl3
import numpy as np
import soundfile as sf
import tensorflow_hub as hub
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from utils.audio_tools import extract_audio_ffmpeg


class AudioAnalyzer:
    def __init__(self, video_path, output_dir="temp", whisper_model_size="tiny",
                 ffmpeg_path="ffmpeg", ffprobe_path="ffprobe", device="cuda"):
        self.video_path = video_path
        self.output_dir = output_dir
        self.audio_path = os.path.join(output_dir, "temp_audio.wav")
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
        self.device = device

        os.makedirs(self.output_dir, exist_ok=True)

        self.openl3_model = None
        self.yamnet_model = None
        self.whisper_model = whisper.load_model(whisper_model_size, device=self.device)

        self.segments = []
        self.segments_short = []

    def extract_audio(self):
        print("🎧 Извлечение аудио из видео...")
        extract_audio_ffmpeg(self.video_path, self.audio_path, ffmpeg_path=self.ffmpeg_path)
        print(f"✅ Аудио сохранено: {self.audio_path}")

    def load_models(self):
        print("📦 Загрузка модели torchopenl3...")
        if self.openl3_model is None:
            self.openl3_model = torchopenl3.models.load_audio_embedding_model(
                input_repr="mel256", content_type="music", embedding_size=6144
            ).to(self.device)
        print("✅ torchopenl3 модель загружена.")

        if self.yamnet_model is None:
            self.yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
            print("✅ YAMNet модель загружена.")

    def extract_openl3_features(self):
        print("🔍 Извлечение аудиопризнаков с помощью torchopenl3...")

        # Создаём копию .wav специально для openl3
        openl3_audio_path = os.path.join(self.output_dir, "temp_audio_openl3.wav")

        # Загружаем и пересохраняем как float32 PCM
        audio, sr = sf.read(self.audio_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float32)
        sf.write(openl3_audio_path, audio, sr, subtype="PCM_16")

        # Повторное чтение (на всякий случай)
        audio, sr = sf.read(openl3_audio_path)
        audio = audio.astype(np.float32)

        print(f"⏱ Длительность аудио: {len(audio) / sr:.2f} сек")

        embeddings, timestamps = torchopenl3.get_audio_embedding(
            audio,
            sr,
            model=self.openl3_model,
            hop_size=1.0,
            center=True
        )

        embeddings = embeddings.squeeze(0)  # (1, T, D) → (T, D)

        print(f"✅ torchopenl3 вернул {embeddings.shape[0]} эмбеддингов.")

        if embeddings.shape[0] > 1:
            embeddings = (embeddings - embeddings.mean(dim=0)) / embeddings.std(dim=0)
        else:
            print("⚠️ Пропущена нормализация — только один эмбеддинг.")

        return embeddings.cpu().numpy(), timestamps

    def extract_yamnet_labels(self):
        print("🔊 Получение меток с YAMNet...")
        waveform, sr = librosa.load(self.audio_path, sr=16000, mono=True)
        waveform = waveform.astype(np.float32)

        scores, embeddings, spectrogram = self.yamnet_model(waveform)

        try:
            predictions = np.argmax(scores.numpy(), axis=1)
        except Exception as e:
            print("❌ Ошибка при обработке YAMNet:", e)
            predictions = []

        print(f"✅ Получено {len(predictions)} аудиометок.")
        return predictions.tolist()

    def compute_rms(self, frame_duration=1.0):
        print("📊 Расчёт RMS-громкости...")
        y, sr = librosa.load(self.audio_path, sr=None)
        frame_length = int(sr * frame_duration)
        hop_length = frame_length
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        print(f"✅ RMS рассчитано для {len(rms)} окон.")
        return times, rms

    def detect_scenes(self, features, timestamps, yamnet_labels, rms_t, rms_vals, sensitivity, min_scene_duration):
        print("✂️ Детектирование границ сцен...")
        if features.shape[0] < 2:
            raise ValueError("Недостаточно эмбеддингов для анализа.")

        smoothed = gaussian_filter1d(features, sigma=2, axis=0)
        diffs = [
            1 - cosine_similarity([smoothed[i - 1]], [smoothed[i]])[0][0]
            for i in range(1, len(smoothed))
        ]
        diffs = np.array(diffs)

        threshold = np.percentile(diffs, 100 * sensitivity)
        change_points = [i for i, d in enumerate(diffs) if d > threshold]

        print(f"🟢 Найдено {len(change_points)} переходов сцен.")
        return change_points

    def process_asr(self, language="ru"):
        print("🗣 Распознавание речи с Whisper...")
        result = self.whisper_model.transcribe(self.audio_path, language=language)
        self.segments = result["segments"]
        self.segments_short = [
            {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
            for seg in result["segments"]
        ]

    def run(self, sensitivity=0.85, min_scene_duration=2.0):
        print("🚀 Запуск аудиоанализа...")
        self.extract_audio()
        self.load_models()

        features, timestamps = self.extract_openl3_features()
        yamnet_labels = self.extract_yamnet_labels()
        rms_t, rms_vals = self.compute_rms()

        changes = self.detect_scenes(features, timestamps, yamnet_labels, rms_t, rms_vals,
                                     sensitivity, min_scene_duration)

        self.process_asr(language="ru")
        print("✅ Анализ завершён.")

    def detect_audio_activity(self, frame_duration=1.0):
        _, rms_vals = self.compute_rms(frame_duration)
        return rms_vals.tolist()
