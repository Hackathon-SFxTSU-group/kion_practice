import whisper # Библиотека модели для распознавания речи от OpenAI
import librosa # Библиотека для загрузки и анализа аудиофайлов
import numpy as np # Для численных операций
import torch # Для работы с GPU/CPU и определения доступности CUDA


class AudioAnalyzer:
    def __init__(self, audio_path, model_size="tiny", device="cuda" if torch.cuda.is_available() else "cpu"):
        # Загружаем модель Whisper заданного размера на доступное устройство (GPU или CPU)
        self.model = whisper.load_model(model_size, device=device)
        self.audio_path = audio_path

    def transcribe_audio(self):
        # Распознаем речь из аудиофайла
        result = self.model.transcribe(self.audio_path)

        # Возвращаем список сегментов с распознанным текстом и временными метками
        return result['segments']

    def detect_audio_activity(self, frame_duration=1.0):
        # Загружаем аудио без изменения частоты дискретизации (sr)
        y, sr = librosa.load(self.audio_path, sr=None)
        # Вычисляем длину одного фрейма (в сэмплах) по заданной длительности (в секундах)
        frame_length = int(sr * frame_duration)
        # Разбиваем аудио на фреймы и вычисляем RMS-энергию для каждого
        # Это позволяет оценить активность (громкость) звука во времени
        energy = [
            np.sqrt(np.mean(y[i:i + frame_length] ** 2)) # Корень из средней квадратичной энергии фрейма
            for i in range(0, len(y), frame_length)
        ]
        # Возвращаем список энергий по фреймам
        return energy
