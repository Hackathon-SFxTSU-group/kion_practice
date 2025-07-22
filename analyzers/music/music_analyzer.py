import os
import numpy as np
import librosa
import openl3
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from moviepy.editor import VideoFileClip
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics.pairwise import cosine_similarity



class MusicAnalyzer:
    def __init__(self, video_path, output_dir="output_scenes"):
        # Инициализация путей, моделей и создание выходной директории
        self.video_path = video_path
        self.output_dir = output_dir
        self.audio_path = "temp_audio.wav"
        self.openl3_model = None
        self.yamnet_model = None
        os.makedirs(output_dir, exist_ok=True)

    def extract_audio(self):
        # Извлечение аудио из видео и сохранение его в WAV-файл
        clip = VideoFileClip(self.video_path)
        clip.audio.write_audiofile(self.audio_path, fps=44100, codec='pcm_s16le', verbose=False, logger=None)
        self.video_duration = clip.duration
        clip.close()

    def load_models(self):
        # Загрузка моделей OpenL3 и YAMNet, если они ещё не загружены
        if self.openl3_model is None:
            self.openl3_model = openl3.models.load_audio_embedding_model("mel256", "music", 6144)
        if self.yamnet_model is None:
            self.yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

    def extract_openl3_features(self):
        # Извлечение аудиоэмбеддингов с помощью модели OpenL3
        y, sr = librosa.load(self.audio_path, sr=44100, mono=True)
        features, timestamps = openl3.get_audio_embedding(y, sr, model=self.openl3_model, hop_size=1.0, center=True, verbose=0)
        features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)  # нормализация
        return features, timestamps

    def extract_yamnet_labels(self):
        # Классификация аудио по типу звуков с помощью модели YAMNet
        waveform, sr = librosa.load(self.audio_path, sr=16000)
        yamnet_output = self.yamnet_model(waveform)
        scores = yamnet_output[0].numpy()  # вероятности классов
        labels = np.argmax(scores, axis=1)  # метки с максимальной вероятностью
        return labels, scores

    def compute_rms(self):
        # Вычисление уровня громкости по RMS (root mean square)
        y, sr = librosa.load(self.audio_path, sr=44100)
        rms = librosa.feature.rms(y=y)[0]
        return librosa.frames_to_time(np.arange(len(rms)), sr=sr), rms

    def detect_scenes(self, features, timestamps, yamnet_labels, rms_t, rms_vals, sensitivity=0.85, min_scene_duration=2.0):
        # Детекция границ сцен на основе изменений в аудиоэмбеддингах, метках YAMNet и RMS
        smoothed = gaussian_filter1d(features, sigma=2, axis=0)  # сглаживание эмбеддингов
        diffs = [1 - cosine_similarity([smoothed[i-1]], [smoothed[i]])[0][0] for i in range(1, len(smoothed))]  # косинусное различие
        threshold = np.percentile(diffs, sensitivity * 100)  # порог отсечения
        changes = []
        prev_time = -min_scene_duration
        self.detection_log = []

        for i, diff in enumerate(diffs):
            t = timestamps[i+1]
            audio_change = diff > threshold
            yamnet_change = (yamnet_labels[i] != yamnet_labels[i+1])
            rms_change = np.abs(rms_vals[min(i, len(rms_vals)-1)] - rms_vals[max(i-1, 0)]) > 0.1
            vote = sum([audio_change, yamnet_change, rms_change]) >= 2  # голосование 2 из 3
            self.detection_log.append((t, audio_change, yamnet_change, rms_change, vote))
            if vote and (t - prev_time >= min_scene_duration):
                changes.append(t)
                prev_time = t

        return changes, diffs, threshold

    def plot_diagnostics(self, timestamps, features, changes, diffs, threshold):
        # Построение графиков (статичных) для анализа: эмбеддинги и различия
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(timestamps, np.mean(features, axis=1))
        for ch in changes:
            plt.axvline(x=ch, color='r', alpha=0.4)
        plt.title("OpenL3 Features with Scene Changes")

        plt.subplot(2, 1, 2)
        plt.plot(timestamps[1:], diffs)
        plt.axhline(y=threshold, color='g', linestyle='--')
        for ch in changes:
            plt.axvline(x=ch, color='r', alpha=0.4)
        plt.title("Cosine Diffs + Threshold")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "scene_analysis.png"))
        plt.close()

    def plot_interactive(self):
        # Создание интерактивного графика сценных признаков в Plotly и сохранение как HTML
        if not hasattr(self, 'detection_log'):
            print("Нет данных для интерактивного графика")
            return

        df = pd.DataFrame(self.detection_log, columns=['time','audio_change','yamnet_change','rms_change','voted'])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['time'], y=df['audio_change'], mode='lines', name='Audio Change'))
        fig.add_trace(go.Scatter(x=df['time'], y=df['yamnet_change'], mode='lines', name='YAMNet Change'))
        fig.add_trace(go.Scatter(x=df['time'], y=df['rms_change'], mode='lines', name='RMS Change'))
        fig.add_trace(go.Scatter(x=df['time'], y=df['voted'], mode='lines', name='Voted Scene Cut', line=dict(width=2)))

        fig.update_layout(title="Scene Detection Signals", xaxis_title="Time (s)", yaxis_title="Binary Flags",
                          template="plotly_white")
        fig.write_html(os.path.join(self.output_dir, "interactive_scene_plot.html"))

    def export_scene_csv(self, changes):
        # Экспорт информации о сценах в CSV-файл
        scene_bounds = [0] + changes + [self.video_duration]
        csv_path = os.path.join(self.output_dir, "scene_timeline.csv")
        with open(csv_path, 'w') as f:
            f.write("scene,start,end,duration\n")
            for i in range(len(scene_bounds)-1):
                start = scene_bounds[i]
                end = scene_bounds[i+1]
                f.write(f"{i+1},{start:.2f},{end:.2f},{(end-start):.2f}\n")

    def export_detection_report(self):
        # Экспорт подробного отчёта по каждому моменту голосования (по трём каналам)
        path = os.path.join(self.output_dir, "detection_report.csv")
        with open(path, 'w') as f:
            f.write("time,audio_change,yamnet_change,rms_change,voted\n")
            for t, a, y, r, v in self.detection_log:
                f.write(f"{t:.2f},{int(a)},{int(y)},{int(r)},{int(v)}\n")

    def split_video_by_scenes(self, changes):
        # Нарезка видео по временным меткам сцен
        clip = VideoFileClip(self.video_path)
        scene_bounds = [0] + changes + [self.video_duration]
        for i in range(len(scene_bounds)-1):
            start = scene_bounds[i]
            end = scene_bounds[i+1]
            subclip = clip.subclip(start, end)
            output_path = os.path.join(self.output_dir, f"scene_{i+1:03d}.mp4")
            subclip.write_videofile(output_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)
        clip.close()

    def run(self, sensitivity=0.85, min_scene_duration=2.0):
        # Главный метод: запускает весь процесс от аудио до нарезки и отчётов
        self.extract_audio()
        self.load_models()
        features, timestamps = self.extract_openl3_features()
        yamnet_labels, yamnet_scores = self.extract_yamnet_labels()
        rms_t, rms_vals = self.compute_rms()
        changes, diffs, threshold = self.detect_scenes(features, timestamps, yamnet_labels, rms_t, rms_vals, sensitivity, min_scene_duration)
        self.plot_diagnostics(timestamps, features, changes, diffs, threshold)
        self.plot_interactive()
        self.export_scene_csv(changes)
        self.export_detection_report()
        self.split_video_by_scenes(changes)
        print(f"✅ Найдено {len(changes)+1} сцен. Сохранены в {self.output_dir}")
