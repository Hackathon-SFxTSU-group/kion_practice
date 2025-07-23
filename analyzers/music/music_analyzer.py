import os
import numpy as np
import librosa
import openl3
import json
import torch
import whisper
import tensorflow_hub as hub
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics.pairwise import cosine_similarity
from utils.audio_tools import extract_audio_ffmpeg, get_video_duration


class AudioSceneAnalyzer:
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
        self.whisper_model = whisper.load_model(
            whisper_model_size, device="cuda" if torch.cuda.is_available() else "cpu"
        )
        os.makedirs(output_dir, exist_ok=True)

    def extract_audio(self):
        self.audio_path = extract_audio_ffmpeg(self.video_path, self.audio_path, self.ffmpeg_path)
        self.video_duration = get_video_duration(self.video_path, self.ffprobe_path)

    def load_models(self):
        if self.openl3_model is None:
            self.openl3_model = openl3.models.load_audio_embedding_model("mel256", "music", 6144)
        if self.yamnet_model is None:
            self.yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

    def extract_openl3_features(self):
        y, sr = librosa.load(self.audio_path, sr=44100, mono=True)
        features, timestamps = openl3.get_audio_embedding(
            y, sr, model=self.openl3_model, hop_size=1.0, center=True, verbose=0
        )
        features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
        return features, timestamps

    def extract_yamnet_labels(self):
        waveform, sr = librosa.load(self.audio_path, sr=16000)
        yamnet_output = self.yamnet_model(waveform)
        scores = yamnet_output[0].numpy()
        labels = np.argmax(scores, axis=1)
        return labels

    def compute_rms(self):
        y, sr = librosa.load(self.audio_path, sr=44100)
        rms = librosa.feature.rms(y=y)[0]
        return librosa.frames_to_time(np.arange(len(rms)), sr=sr), rms

    def detect_scenes(self, features, timestamps, yamnet_labels, rms_t, rms_vals,
                      sensitivity=0.85, min_scene_duration=2.0):
        smoothed = gaussian_filter1d(features, sigma=2, axis=0)
        diffs = [1 - cosine_similarity([smoothed[i - 1]], [smoothed[i]])[0][0] for i in range(1, len(smoothed))]
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

        return changes

    def transcribe_audio(self):
        result = self.whisper_model.transcribe(self.audio_path)
        return result['segments']

    def detect_audio_activity(self, frame_duration=1.0):
        y, sr = librosa.load(self.audio_path, sr=None)
        frame_length = int(sr * frame_duration)
        energy = [
            np.sqrt(np.mean(y[i:i + frame_length] ** 2))
            for i in range(0, len(y), frame_length)
        ]
        return energy

    def export_all_to_json(self, changes, transcript,
                           method_name="audio_openl3_yamnet_rms", sensitivity=0.85, min_scene_duration=2.0):
        scene_bounds = [0] + changes + [self.video_duration]
        scenes = []
        for i in range(len(scene_bounds) - 1):
            start = scene_bounds[i]
            end = scene_bounds[i + 1]
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

        json_data = {
            "method": method_name,
            "scenes": scenes,
            "detection_log": detection_log,
            "speech_segments": transcript,
            "meta": {
                "video_path": self.video_path,
                "model_used": ["openl3", "yamnet", "rms", "whisper"],
                "sensitivity": sensitivity,
                "min_scene_duration": min_scene_duration
            }
        }

        json_path = os.path.join(self.output_dir, f"{method_name}_scenes_full.json")
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=4)

        print(f"✅ Полный отчёт сохранён в {json_path}")

    def run(self, sensitivity=0.85, min_scene_duration=2.0):
        self.extract_audio()
        self.load_models()
        features, timestamps = self.extract_openl3_features()
        yamnet_labels = self.extract_yamnet_labels()
        rms_t, rms_vals = self.compute_rms()
        changes = self.detect_scenes(features, timestamps, yamnet_labels, rms_t, rms_vals,
                                     sensitivity, min_scene_duration)
        transcript = self.transcribe_audio()
        self.export_all_to_json(changes, transcript, sensitivity=sensitivity, min_scene_duration=min_scene_duration)
        print(f"✅ Найдено {len(changes)+1} сцен. Добавлено {len(transcript)} фрагментов речи.")
