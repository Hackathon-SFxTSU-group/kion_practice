# ============================ ИМПОРТЫ ============================
import os
import pickle
import matplotlib.pyplot as plt

# Локальные модули (обрати внимание на актуальные пути)
# from classes.video_analyzer_complex import VideoPipeline
from classes.video_anlyzer_without_faceraid import VideoPipeline
from classes.scene_enricher import SceneEnricher
from classes.openai import OpenAIThemer
from classes.scene_merge import UnifiedSceneMerger
from classes.audioanalyzer_and_transcriptor import AudioAnalyzer

# ============================ ПАРАМЕТРЫ ============================
VIDEO_PATH = "videos/Video_02.avi"           # Путь к видеофайлу
AUDIO_PATH = "temp/temp_audio.wav"           # Временный WAV-файл
OUTPUT_DIR = "output_files/"                 # Папка для промежуточных выходных файлов
TEMP_DIR = "temp/"                           # Папка для временных данных
CACHE_PATH = os.path.join(TEMP_DIR, "results.pkl")  # Кеш-файл с результатами видеоанализа

MIN_SCENE_LENGTH = 2.0      # Минимальная длина сцены (сек)
MAX_SCENE_LENGTH = 300.0    # Максимальная длина сцены (сек)

# ============================ ВИДЕОАНАЛИЗ ============================

if os.path.exists(CACHE_PATH):
    print("🔁 Загружаем результаты видеоанализа из кеша...")
    with open(CACHE_PATH, "rb") as f:
        # Загружаем сцены, лица и трекинг-координаты
        scenes, track_faces, tracking_frames = pickle.load(f)
else:
    print("📽 Запуск видеоанализа...")
    pipeline = VideoPipeline()
    scenes, track_faces, tracking_frames = pipeline.run(VIDEO_PATH, OUTPUT_DIR)

    # Сохраняем в кеш для повторного использования
    os.makedirs(TEMP_DIR, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump((scenes, track_faces, tracking_frames), f)

# ============================ АУДИОАНАЛИЗ ============================

print("🎧 Запуск аудиоанализа...")
splitter = AudioAnalyzer(
    video_path=VIDEO_PATH,
    output_dir=OUTPUT_DIR,
    whisper_model_size='small'
)
splitter.run(sensitivity=0.88, min_scene_duration=MIN_SCENE_LENGTH)

# Краткие и полные сегменты речи
segments_short = splitter.segments_short   # [{start, end, text}]
segments_full = splitter.segments          # Whisper-raw segments

# Расчёт аудиоэнергии для сцен
print("📈 Вычисление аудиоэнергии...")
energy = splitter.detect_audio_activity(frame_duration=1.0)

# ============================ ОБОГАЩЕНИЕ СЦЕН ============================

print("🧠 Обогащение сцен данными из аудио и текста...")
enricher = SceneEnricher(segments_full, energy)
# Если есть `track_id_to_person`, передать его вторым аргументом
scenes_final = enricher.run(scenes)

# ============================ ТЕМАТИЧЕСКАЯ АНАЛИТИКА ============================

print("💬 Запуск тематической группировки через LLM...")
themer = OpenAIThemer(
    api_key="",  # 🔐 Укажи свой OpenAI API ключ
    base_url="https://api.proxyapi.ru/openai/v1"
)
themes = themer.get_themes(segments_short, audio_path=VIDEO_PATH)

# ============================ ОБЪЕДИНЕНИЕ СЦЕН ============================

print("🧩 Финальное объединение сцен...")
merger = UnifiedSceneMerger(VIDEO_PATH)
final_scenes = merger.run()

# ============================ ГОТОВО ============================

print(f"\n✅ Анализ завершён. Сцен после объединения: {len(final_scenes)}")
