import os
import re
import json
import numpy as np
from sentence_transformers import util
from sklearn.cluster import DBSCAN

# ------------------------- Персонажи и идентификаторы -------------------------

def enrich_scenes_with_characters(scene_data, track_id_to_person):
    """
    Добавляет список персонажей в каждую сцену на основе отображения track_id → имя персонажа.
    """
    for scene in scene_data:
        scene['characters'] = list({
            track_id_to_person.get(tid)
            for tid in scene['track_ids']
            if tid in track_id_to_person
        })
    return scene_data


def get_identities(scene):
    """
    Получает список 'characters' или 'track_ids' из сцены.
    Используется для сравнения идентичности между сценами.
    """
    return scene.get('characters', []) or scene.get('track_ids', [])


# ------------------------- Текстовые функции -------------------------

def normalize_text(t):
    """
    Приводит текст к нижнему регистру и убирает пунктуацию.
    """
    return re.sub(r'[^\w\s]', '', t.lower()).strip()


def get_text_score(a, b, sentenceTransformer):
    """
    Вычисляет семантическое сходство между двумя строками с помощью SentenceTransformer.
    """
    a, b = normalize_text(a), normalize_text(b)
    if not a or not b:
        return 0.0
    a_embed = sentenceTransformer.encode(a, convert_to_tensor=True)
    b_embed = sentenceTransformer.encode(b, convert_to_tensor=True)
    return float(util.cos_sim(a_embed, b_embed)[0][0])


# ------------------------- Группировка сцен -------------------------

def jaccard_similarity(a, b):
    """
    Вычисляет коэффициент Жаккара между двумя множествами.
    Используется для оценки схожести персонажей.
    """
    set_a, set_b = set(a), set(b)
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0


def group_semantic_scenes(scene_data, sentenceTransformer, char_thresh=0.5, text_thresh=0.55, audio_thresh=0.02):
    """
    Объединяет соседние сцены, которые схожи по:
    - составу персонажей,
    - тексту (семантически),
    - уровню громкости (RMS).
    """
    grouped = []
    buffer = [scene_data[0]]
    prev_text = scene_data[0]['transcript']

    for curr in scene_data[1:]:
        last = buffer[-1]

        char_score = jaccard_similarity(get_identities(curr), get_identities(last))
        text_score = get_text_score(prev_text, curr['transcript'], sentenceTransformer)
        audio_diff = abs(curr['avg_rms'] - last['avg_rms'])

        if char_score >= char_thresh and text_score >= text_thresh and audio_diff <= audio_thresh:
            buffer.append(curr)
            prev_text += " " + curr['transcript']
        else:
            grouped.append(_build_scene_from_buffer(buffer))
            buffer = [curr]
            prev_text = curr['transcript']

    if buffer:
        grouped.append(_build_scene_from_buffer(buffer))
    return grouped


def _build_scene_from_buffer(buffer):
    """
    Собирает финальную сцену из нескольких объединённых.
    """
    return {
        'start': buffer[0]['start'],
        'end': buffer[-1]['end'],
        'characters': sorted(set().union(*(get_identities(b) for b in buffer))),
        'transcript': " ".join(b['transcript'] for b in buffer),
        'avg_rms': float(np.mean([b['avg_rms'] for b in buffer]))
    }


def cluster_scenes_with_time_windows(scenes, sentenceTransformer, window_size=60, eps=0.45, min_samples=2):
    """
    Выполняет кластеризацию сцен в пределах временных окон.
    Использует DBSCAN по семантическим эмбеддингам текста.
    """
    max_time = max(s['end'] for s in scenes)
    scenes_sorted = sorted(scenes, key=lambda x: x['start'])
    chapters = []
    start_window = 0

    while start_window < max_time:
        end_window = start_window + window_size
        window_scenes = [s for s in scenes_sorted if start_window <= s['start'] < end_window]
        if not window_scenes:
            start_window = end_window
            continue

        transcripts = [s['transcript'] for s in window_scenes]
        embeddings = sentenceTransformer.encode(transcripts, convert_to_tensor=False, normalize_embeddings=True)
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = clustering.fit_predict(embeddings)

        clusters = {}
        for label, scene in zip(labels, window_scenes):
            label = f"scene_{scene['start']:.2f}" if label == -1 else label
            if label not in clusters:
                clusters[label] = {
                    "start": scene["start"],
                    "end": scene["end"],
                    "characters": set(scene["characters"]),
                    "transcripts": [scene["transcript"]],
                    "avg_rms": [scene["avg_rms"]],
                }
            else:
                clusters[label]["end"] = max(clusters[label]["end"], scene["end"])
                clusters[label]["characters"].update(scene["characters"])
                clusters[label]["transcripts"].append(scene["transcript"])
                clusters[label]["avg_rms"].append(scene["avg_rms"])

        for cluster in clusters.values():
            chapters.append({
                "start": cluster["start"],
                "end": cluster["end"],
                "characters": sorted(cluster["characters"]),
                "transcript": " ".join(cluster["transcripts"]),
                "avg_rms": float(np.mean(cluster["avg_rms"]))
            })

        start_window = end_window

    return sorted(chapters, key=lambda x: x['start'])


# ------------------------- Обогащение и фильтрация -------------------------

def enrich_scenes_with_audio(scenes, segments, energy, frame_duration=1.0):
    """
    Добавляет в сцены:
    - текст по распознанной речи,
    - среднюю RMS-энергию по кадрам.
    """
    for scene in scenes:
        start, end = scene["start"], scene["end"]

        # Привязка текста
        scene_texts = [
            clip_text_to_scene(seg, start, end)
            for seg in segments
            if not (seg["end"] < start or seg["start"] > end)
        ]
        scene["transcript"] = " ".join(filter(None, scene_texts)).strip()

        # Расчёт энергии
        start_idx = max(0, int(start // frame_duration))
        end_idx = min(len(energy) - 1, int(end // frame_duration))
        scene_energy = energy[start_idx:end_idx + 1] if end_idx >= start_idx else []
        scene["avg_rms"] = float(np.mean(scene_energy)) if scene_energy else 0.0

    return scenes


def clean_and_merge_short_scenes(scenes, min_duration=2.0, min_words=3):
    """
    Удаляет короткие и малосодержательные сцены, сливая их с предыдущими.
    """
    cleaned = []
    buffer = None

    for scene in scenes:
        duration = scene["end"] - scene["start"]
        word_count = len(scene["transcript"].strip().split())

        if word_count < min_words or duration < 0.5:
            if cleaned:
                cleaned[-1] = merge_scene(cleaned[-1], scene)
            continue

        if duration < min_duration:
            if cleaned:
                cleaned[-1] = merge_scene(cleaned[-1], scene)
            else:
                buffer = scene
        else:
            cleaned.append(scene)

    if buffer:
        if cleaned:
            cleaned[-1] = merge_scene(cleaned[-1], buffer)
        else:
            cleaned.append(buffer)

    return cleaned


def merge_scene(a, b):
    """
    Объединяет две сцены в одну.
    """
    return {
        "start": a["start"],
        "end": b["end"],
        "characters": sorted(set(a.get("characters", []) + b.get("characters", []))),
        "transcript": " ".join([a["transcript"], b["transcript"]]).strip(),
        "avg_rms": float(np.mean([a["avg_rms"], b["avg_rms"]]))
    }


def clip_text_to_scene(seg, start, end):
    """
    Вырезает релевантную часть текста из сегмента, попадающую в сцену.
    """
    seg_start, seg_end = seg["start"], seg["end"]
    overlap_start = max(start, seg_start)
    overlap_end = min(end, seg_end)
    overlap_dur = max(0.0, overlap_end - overlap_start)
    total_dur = seg_end - seg_start

    if overlap_dur == 0 or total_dur == 0:
        return ""

    words = seg["text"].strip().split()
    if len(words) <= 2:
        return seg["text"].strip()

    ratio = overlap_dur / total_dur
    count = max(1, int(len(words) * ratio))
    return " ".join(words[:count]) if overlap_start == seg_start else " ".join(words[-count:])


# ------------------------- Временные коррекции и экспорт -------------------------

def resolve_time_overlaps(chapters):
    """
    Удаляет перекрывающиеся сцены, сдвигая начала, чтобы избежать наложений.
    """
    chapters.sort(key=lambda x: x['start'])
    resolved = []
    prev_end = 0

    for ch in chapters:
        start = max(ch['start'], prev_end)
        end = ch['end']
        if start >= end:
            continue
        resolved.append({**ch, 'start': start, 'end': end})
        prev_end = end

    return resolved


def save_scenes_report_to_json(scenes):
    """
    Сохраняет краткий JSON-отчет о сценах в папку `reports/`.
    """
    os.makedirs("reports", exist_ok=True)
    base_name = "scenes_enriched"
    output_path = os.path.join("reports", f"{base_name}_report.json")

    scene_list = []
    for i, scene in enumerate(scenes):
        scene_info = {
            "scene_id": i + 1,
            "start_time_sec": round(scene["start"], 2),
            "end_time_sec": round(scene["end"], 2),
            "characters": scene.get("characters", []),
            "avg_rms": round(scene.get("avg_rms", 0.0), 4),
            "transcript": scene.get("transcript", "").strip()
        }
        scene_list.append(scene_info)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scene_list, f, ensure_ascii=False, indent=4)

    print(f"✅ JSON-отчёт по сценам сохранён: {output_path}")


def print_scenes_formatted(scenes):
    """
    Печатает сцены в человекочитаемом формате с временными метками и диалогами.
    """
    for i, scene in enumerate(scenes):
        print(f"\n🎬 Сцена {i + 1}")
        print(f"⏱ Время: {scene['start']:.2f} – {scene['end']:.2f} сек")
        print(f"🧍 Персонажи: {', '.join(scene['characters'])}")
        print(f"🔊 Средняя громкость: {scene['avg_rms']:.4f}")
        print(f"🗣 Реплика:\n\"{scene['transcript'].strip()}\"")
        print("-" * 60)
