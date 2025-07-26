import os
import torch
import torchaudio
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics.pairwise import cosine_similarity
import whisper
from panns_inference import AudioTagging
import subprocess


# 📁 Вспомогательные функции
def extract_audio_ffmpeg(video_path, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1", out_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path

def get_video_duration(video_path):
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    return n_frames / fps


# 🎧 Класс-анализатор
class AudioSceneAnalyzer:
    def __init__(self, video_path, output_dir="output_scenes", device=None):
        self.video_path = video_path
        self.output_dir = output_dir
        self.audio_path = os.path.join(output_dir, "temp_audio.wav")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(output_dir, exist_ok=True)

        # Whisper
        self.whisper_model = whisper.load_model("small", device=self.device)

        # PANNs (Cnn14)
        self.panns_model = AudioTagging(device=self.device)
        self.target_sr = 32000

    def extract_audio(self):
        self.audio_path = extract_audio_ffmpeg(self.video_path, self.audio_path)
        self.video_duration = get_video_duration(self.video_path)

    def extract_panns_features(self, window_sec=1.0):
        waveform, sr = torchaudio.load(self.audio_path)
        waveform = waveform.mean(dim=0)  # моно

        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)

        step_size = int(self.target_sr * window_sec)
        features = []
        timestamps = []

        for i in range(0, len(waveform) - step_size, step_size):
            chunk = waveform[i:i + step_size].unsqueeze(0).to(self.device)
            if chunk.shape[-1] < step_size:
                continue

            with torch.no_grad():
                _, emb = self.panns_model.inference(chunk)  # emb: numpy array
                emb = emb.squeeze()                         # remove singleton dimensions
                emb = emb / np.linalg.norm(emb)
                features.append(emb)
                timestamps.append(i / self.target_sr)

        features = np.array(features)
        return features, timestamps


    def detect_scenes(self, features, timestamps, sensitivity=0.85):
        smoothed = gaussian_filter1d(features, sigma=1, axis=0)
        diffs = [1 - cosine_similarity([smoothed[i - 1]], [smoothed[i]])[0][0]
                 for i in range(1, len(smoothed))]
        threshold = np.percentile(diffs, sensitivity * 100)
        cuts = [timestamps[i + 1] for i, d in enumerate(diffs) if d > threshold]
        return cuts, diffs, timestamps[1:]

    def transcribe(self):
        result = self.whisper_model.transcribe(self.audio_path)
        return result['segments']
