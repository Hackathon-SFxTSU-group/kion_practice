import subprocess
import os

class VideoSceneDetector:
    def __init__(self, video_path, threshold=30.0):
        self.video_path = video_path
        self.threshold = threshold  # PySceneDetect threshold

    def detect_scenes(self):
        # Используем scenedetect через команду
        output_file = "video_scenes.csv"
        cmd = f"scenedetect -i \"{self.video_path}\" detect-content -t {self.threshold} list-scenes -o ."
        subprocess.run(cmd, shell=True, check=True)

        # Парсим CSV
        cuts = []
        with open(output_file, 'r') as f:
            for line in f:
                if line.startswith('Scene Number'):
                    continue
                parts = line.strip().split(',')
                if len(parts) > 3:
                    timecode = parts[3].strip()
                    h, m, s = map(float, timecode.split(':'))
                    t = h*3600 + m*60 + s
                    cuts.append(t)
        os.remove(output_file)
        return cuts
