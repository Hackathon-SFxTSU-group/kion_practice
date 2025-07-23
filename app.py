# Импорт необходимых модулей
import os
import torch
import pickle

from utils.video_tools import save_scenes_ffmpeg
from classes.audio_analyzer import AudioSceneAnalyzer
from classes.video_analyzer_complex import VideoPipeline
from classes.scene_enricher import SceneEnricher
from classes.simple_transcriptor import ASRProcessor
from classes.openai import OpenAIThemer
from classes.scene_merger import UnifiedSceneMerger


VIDEO_PATH = "videos/Video_01.mp4"          # Путь к видеофайлу
AUDIO_PATH = "temp/temp_audio.wav"           # Временный WAV
OUTPUT_DIR = "output_files/"
TEMP_DIR = "temp/"         # Папка для результатов
CACHE_PATH = os.path.join(TEMP_DIR, "results.pkl")
MIN_SCENE_LENGTH = 2.0                  # Мин. длительность сцены
MAX_SCENE_LENGTH = 300.0                # Макс. допустимая длина сцены (5 мин)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


if os.path.exists(CACHE_PATH):
    print("🔁 Загружаем из кеша...")
    with open(CACHE_PATH, "rb") as f:
        scenes, track_faces, tracking_frames, track_id_to_person = pickle.load(f)

else:
    pipeline = VideoPipeline()
    scenes, track_faces, tracking_frames, track_id_to_person = pipeline.run(VIDEO_PATH, OUTPUT_DIR)

    with open(CACHE_PATH, "wb") as f:
        pickle.dump((scenes, track_faces, tracking_frames, track_id_to_person), f)

splitter = AudioSceneAnalyzer(
    video_path=VIDEO_PATH,
    output_dir=OUTPUT_DIR,
    whisper_model_size='small'
)
splitter.run(sensitivity=0.88, min_scene_duration=2.0)
segments = splitter.transcribe_audio()
energy = splitter.detect_audio_activity(frame_duration=1.0)
enricher = SceneEnricher(segments, energy)
scenes_final = enricher.run(scenes, track_id_to_person)
#scenes_final = enricher.run(scene_data, track_id_to_person, print_report=False)

asr = ASRProcessor()
segments_short = asr.process(VIDEO_PATH)
themer = OpenAIThemer(
        api_key="",  # Добавь свой ключ
        base_url="https://api.proxyapi.ru/openai/v1"
    )
themes = themer.get_themes(segments_short, audio_path=VIDEO_PATH)

merger = UnifiedSceneMerger("video.mp4")
final_scenes = merger.run()