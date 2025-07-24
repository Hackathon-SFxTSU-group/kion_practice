# Импорт необходимых модулей
import os
import pickle
#from classes.video_analyzer_complex import VideoPipeline
from classes.video_anlyzer_without_faceraid import VideoPipeline
from classes.scene_enricher import SceneEnricher
from classes.openai import OpenAIThemer
from classes.simple_transcriptor import ASRProcessor
from classes.scene_merge import UnifiedSceneMerger
import matplotlib.pyplot as plt
from classes.audioanalyzer_and_transcriptor import AudioAnalyzer


VIDEO_PATH = "videos/Video_02.avi"          # Путь к видеофайлу
AUDIO_PATH = "temp/temp_audio.wav"           # Временный WAV
OUTPUT_DIR = "output_files/"
TEMP_DIR = "temp/"         # Папка для результатов
CACHE_PATH = os.path.join(TEMP_DIR, "results.pkl")
MIN_SCENE_LENGTH = 2.0                  # Мин. длительность сцены
MAX_SCENE_LENGTH = 300.0                # Макс. допустимая длина сцены (5 мин)

if os.path.exists(CACHE_PATH):
    print("🔁 Загружаем из кеша...")
    with open(CACHE_PATH, "rb") as f:
        #scenes, track_faces, tracking_frames, track_id_to_person = pickle.load(f)
        scenes, track_faces, tracking_frames = pickle.load(f)

else:
    pipeline = VideoPipeline()
    #scenes, track_faces, tracking_frames, track_id_to_person = pipeline.run(VIDEO_PATH, OUTPUT_DIR)
    scenes, track_faces, tracking_frames = pipeline.run(VIDEO_PATH, OUTPUT_DIR)

    with open(CACHE_PATH, "wb") as f:
        #pickle.dump((scenes, track_faces, tracking_frames, track_id_to_person), f)
        pickle.dump((scenes, track_faces, tracking_frames), f)

# Запуск анализа
splitter = AudioAnalyzer(
    video_path=VIDEO_PATH,
    output_dir=OUTPUT_DIR,
    whisper_model_size='small'
)
splitter.run(sensitivity=0.88, min_scene_duration=2.0)

segments_short = splitter.segments_short  # для отчётов и группировок
segments_full = splitter.segments   

# Вычисление аудиоэнергии по кадрам
energy = splitter.detect_audio_activity(frame_duration=1.0)
enricher = SceneEnricher(segments_full, energy)
#scenes_final = enricher.run(scenes, track_id_to_person)
scenes_final = enricher.run(scenes)
#scenes_final = enricher.run(scene_data, track_id_to_person, print_report=False)

themer = OpenAIThemer(
        api_key="",  # Добавь свой ключ
        base_url="https://api.proxyapi.ru/openai/v1"
    )
themes = themer.get_themes(segments_short, audio_path=VIDEO_PATH)

merger = UnifiedSceneMerger("video.mp4")
final_scenes = merger.run()