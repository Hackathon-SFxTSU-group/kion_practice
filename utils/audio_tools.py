import subprocess  # Для выполнения внешней команды ffmpeg через Python

def extract_audio_ffmpeg(video_path, audio_path="audio.wav"):
    # Формируем команду ffmpeg для извлечения аудиодорожки из видео
    command = [
        "ffmpeg",
        "-y",  # Перезаписать файл, если он уже существует
        "-i", video_path,  # Входной видеофайл
        "-vn",  # Отключить видео (только аудио)
        "-acodec", "pcm_s16le",  # Кодек: несжатый WAV (16 бит, little endian)
        "-ar", "16000",  # Частота дискретизации 16 кГц — рекомендуемая для Whisper
        "-ac", "1",  # Один аудиоканал (моно)
        audio_path  # Выходной файл аудио
    ]

    # Запускаем команду и проверяем, что она завершилась успешно
    subprocess.run(command, check=True)

    # Возвращаем путь к полученному аудиофайлу
    return audio_path
