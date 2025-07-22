# Импорт необходимых модулей
from analyzers.audio.audio_analyzer import AudioAnalyzer
from tracking.face_reid import FaceReId
from utils.scene_tools import (
   enrich_scenes_with_characters,
   enrich_scenes_with_audio,
   clean_and_merge_short_scenes,
   print_scenes_formatted, cluster_scenes_with_time_windows, resolve_time_overlaps,
)
from utils.audio_tools import extract_audio_ffmpeg
from sentence_transformers import SentenceTransformer

from utils.video_tools import save_scenes_ffmpeg
from analyzers.video.video_analyzer import VideoAnalyzer
from analyzers.music.music_analyzer import MusicAnalyzer
import os
from analyzers.video.render_video import render_video_with_faces

VIDEO_PATH = "videos/Video_01.mp4"          # Путь к видеофайлу
AUDIO_PATH = "temp/temp_audio.wav"           # Временный WAV
OUTPUT_DIR = "splitted_scenes/"         # Папка для результатов
MIN_SCENE_LENGTH = 2.0                  # Мин. длительность сцены
MAX_SCENE_LENGTH = 300.0   

splitter = MusicAnalyzer(
    video_path=VIDEO_PATH,
    output_dir=OUTPUT_DIR
)
splitter.run(sensitivity=0.88, min_scene_duration=2.0)


video_analyzer = VideoAnalyzer()
scenes, track_faces, tracking_frames = video_analyzer.analyze_video(VIDEO_PATH)


face_reid = FaceReId()
track_id_to_person = face_reid.analyze_persons(track_faces)


render_video_with_faces(VIDEO_PATH, "output_with_faces.mp4", tracking_frames, track_id_to_person)


scene_data = enrich_scenes_with_characters(scenes, track_id_to_person)


audio_path = extract_audio_ffmpeg(VIDEO_PATH)


audio_analyzer = AudioAnalyzer(AUDIO_PATH , model_size='small')
segments = audio_analyzer.transcribe_audio()
energy = audio_analyzer.detect_audio_activity(frame_duration=1.0)


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