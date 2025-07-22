import subprocess  # Для запуска ffmpeg через командную строку
import os  # Для создания папки и работы с путями

def save_scenes_ffmpeg(video_path, scenes, output_dir):
    # Создаём выходную директорию, если она ещё не существует
    os.makedirs(output_dir, exist_ok=True)

    # Проходим по списку сцен (каждая сцена — словарь с "start" и "end")
    for idx, scene in enumerate(scenes):
        start = scene["start"]  # Время начала сцены в секундах
        duration = scene["end"] - scene["start"]  # Длительность сцены
        output_path = f"{output_dir}/scene_{idx}.mp4"  # Путь для сохранения сцены

        # Формируем команду ffmpeg для обрезки видео по времени
        cmd = [
            "ffmpeg",
            "-y",  # Перезаписать файл, если он уже существует
            "-i", video_path,  # Входной видеофайл
            "-ss", str(start),  # Время начала сцены
            "-t", str(duration),  # Длительность сцены
            "-c:v", "h264_nvenc",  # Аппаратное ускорение NVIDIA (если доступно)
            "-c:a", "aac",  # Аудиокодек AAC (стандартный)
            output_path  # Имя выходного файла
        ]

        # Запускаем ffmpeg с заданной командой
        subprocess.run(cmd)
