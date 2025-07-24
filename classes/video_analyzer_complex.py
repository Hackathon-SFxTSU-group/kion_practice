import os
import cv2
import json
import time
from tqdm import tqdm
from scene_splitter.clip_scene_splitter import CLIPSceneSplitter
from tracking.detector import ObjectDetector
from tracking.tracker import DeepSortTracker
from tracking.face_reid import FaceReId


class VideoPipeline:
    def __init__(self, using_cache=False, render_video=False):
        self.render_video = render_video
        if not using_cache:
            self.splitter = CLIPSceneSplitter()
            self.detector = ObjectDetector()
            self.tracker = DeepSortTracker()
            self.face_reid = FaceReId()
        self.scene_data = []
        self.track_faces = {}
        self.tracking_frames = {}

    def run(self, video_path, output_dir="output_files"):
        os.makedirs(output_dir, exist_ok=True)
        output_video_name = os.path.splitext(os.path.basename(video_path))[0] + "_annotated.mp4"
        output_video_path = os.path.join(output_dir, output_video_name)

        t0 = time.time()
        scenes = self.splitter.detect_scenes(video_path)
        scene_split_time = time.time() - t0

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        t1 = time.time()
        for (start, end) in tqdm(scenes, desc="Analyzing scenes"):
            cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
            frame_idx = int(start * fps)
            end_idx = int(end * fps)
            track_ids_in_scene = set()

            while frame_idx < end_idx:
                ret, frame = cap.read()
                if not ret:
                    break

                detections = self.detector.detect(frame)
                tracks = self.tracker.update(detections, frame)

                for t in tracks:
                    track_id = t['track_id']
                    x1, y1, x2, y2 = map(int, t['bbox'])
                    track_ids_in_scene.add(track_id)

                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size == 0:
                        continue

                    if track_id not in self.track_faces:
                        self.track_faces[track_id] = []
                    self.track_faces[track_id].append(face_crop)

                    if track_id not in self.tracking_frames:
                        self.tracking_frames[track_id] = {}
                    self.tracking_frames[track_id][frame_idx] = (x1, y1, x2, y2)

                frame_idx += 1

            self.scene_data.append({
                "start": start,
                "end": end,
                "track_ids": list(track_ids_in_scene)
            })
        tracking_time = time.time() - t1
        cap.release()

        self.export_report_to_json(video_path, output_dir, scene_split_time, tracking_time)
        track_id_to_person = self.face_reid.analyze_persons(self.track_faces)

        if self.render_video:
            self.render_labeled_video(video_path, output_video_path, self.tracking_frames, track_id_to_person)

        return self.scene_data, self.track_faces, self.tracking_frames, track_id_to_person

    def export_report_to_json(self, video_path, output_dir, scene_split_time, tracking_time):
        report = {
            "video_path": video_path,
            "scene_data": self.scene_data,
            "track_faces": {str(k): len(v) for k, v in self.track_faces.items()},
            "tracking_frames": {str(k): list(v.keys()) for k, v in self.tracking_frames.items()},
            "timing": {
                "scene_split_time_sec": round(scene_split_time, 3),
                "tracking_time_sec": round(tracking_time, 3)
            }
        }

        os.makedirs("report", exist_ok=True)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join("reports", f"{video_name}_video_report.json")

        with open(output_path, "w") as f:
            json.dump(report, f, indent=4)

        print(f"✅ Отчёт сохранён: {output_path}")

    def render_labeled_video(self, video_path, output_path, tracking_frames, track_id_to_person):
        cap = cv2.VideoCapture(video_path)
        fps = cv2.CAP_PROP_FPS
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            for track_id, frames in tracking_frames.items():
                if frame_idx in frames:
                    x1, y1, x2, y2 = frames[frame_idx]
                    name = track_id_to_person.get(track_id, f"person_{track_id}")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()
        print(f"📼 Размеченное видео сохранено: {output_path}")
