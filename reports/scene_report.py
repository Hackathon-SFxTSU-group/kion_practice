import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

class SceneReportGenerator:
    def __init__(self, json_path):
        self.json_path = json_path
        self.output_dir = os.path.dirname(json_path)
        self.data = self._load_json()

    def _load_json(self):
        with open(self.json_path, 'r') as f:
            return json.load(f)

    def plot_static_diagnostics(self):
        detection_log = self.data["detection_log"]
        df = pd.DataFrame(detection_log)

        voted_times = [row["time"] for row in detection_log if row["voted"]]
        audio_scores = df["audio_change"].astype(int)
        yamnet_scores = df["yamnet_change"].astype(int)
        rms_scores = df["rms_change"].astype(int)

        plt.figure(figsize=(12, 6))

        # Dummy "feature" plot
        plt.subplot(2, 1, 1)
        plt.plot(df["time"], (audio_scores + yamnet_scores + rms_scores) / 3.0, label="Average Change Signal")
        for t in voted_times:
            plt.axvline(x=t, color='r', alpha=0.4)
        plt.title("Voting Signal Strength & Scene Changes")

        # Vote line
        plt.subplot(2, 1, 2)
        plt.plot(df["time"], df["voted"].astype(int), label="Voted", color="black")
        for t in voted_times:
            plt.axvline(x=t, color='r', alpha=0.4)
        plt.title("Voted Scene Cuts")

        plt.tight_layout()
        filename = os.path.join(self.output_dir, f"{self.data['method']}_diagnostics.png")
        plt.savefig(filename)
        plt.close()
        print(f"✅ PNG сохранён: {filename}")

    def plot_interactive(self):
        df = pd.DataFrame(self.data["detection_log"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['time'], y=df['audio_change'], mode='lines', name='Audio Change'))
        fig.add_trace(go.Scatter(x=df['time'], y=df['yamnet_change'], mode='lines', name='YAMNet Change'))
        fig.add_trace(go.Scatter(x=df['time'], y=df['rms_change'], mode='lines', name='RMS Change'))
        fig.add_trace(go.Scatter(x=df['time'], y=df['voted'], mode='lines', name='Voted Scene Cut', line=dict(width=2)))

        fig.update_layout(
            title=f"Scene Detection Signals ({self.data['method']})",
            xaxis_title="Time (s)",
            yaxis_title="Binary Flags",
            template="plotly_white"
        )

        filename = os.path.join(self.output_dir, f"{self.data['method']}_interactive.html")
        fig.write_html(filename)
        print(f"✅ HTML сохранён: {filename}")
