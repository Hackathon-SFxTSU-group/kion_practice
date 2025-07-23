# Импорт необходимых модулей
import os
import torch
import pickle

from utils.scene_tools import (
   enrich_scenes_with_characters,
   enrich_scenes_with_audio,
   clean_and_merge_short_scenes,
   print_scenes_formatted, cluster_scenes_with_time_windows, resolve_time_overlaps,
)

from sentence_transformers import SentenceTransformer
from utils.video_tools import save_scenes_ffmpeg
from analyzers.audio_analyzer import AudioSceneAnalyzer
from analyzers.video_analyzer_complex import VideoPipeline


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


scene_data = enrich_scenes_with_characters(scenes, track_id_to_person)

splitter = AudioSceneAnalyzer(
    video_path=VIDEO_PATH,
    output_dir=OUTPUT_DIR,
    whisper_model_size='small'
)
splitter.run(sensitivity=0.88, min_scene_duration=2.0)
segments = splitter.transcribe_audio()
energy = splitter.detect_audio_activity(frame_duration=1.0)
scene_data_copy = scene_data.copy()
scenes_enrich = enrich_scenes_with_audio(scene_data_copy, segments, energy)
scenes_enrich_copy = scenes_enrich.copy()
sentenceTransformer = SentenceTransformer('all-mpnet-base-v2', device="cuda")
scenes_grouped = cluster_scenes_with_time_windows(scenes_enrich_copy, sentenceTransformer)
grouped_chapters_no_overlap = resolve_time_overlaps(scenes_grouped)
scenes_cleaned = clean_and_merge_short_scenes(grouped_chapters_no_overlap, min_duration=2.0, min_words=3)


print_scenes_formatted(scenes_cleaned)


output_dir = "output_scenes"
save_scenes_ffmpeg(VIDEO_PATH, scenes_cleaned, output_dir)