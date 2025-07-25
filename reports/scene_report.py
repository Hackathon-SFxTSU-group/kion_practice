import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go


class SceneReportGenerator:
    """
    Генератор визуальных отчетов по сценовому аудиоанализу.
    Работает с JSON-файлом, содержащим логи голосования и параметры сцен.
    """

    def __init__(self, json_path):
        """
        Инициализация генератора.

        :param json_path: Путь к JSON-файлу с полным отчетом (от AudioAnalyzer)
        """
        self.json_path = json_path
        self.output_dir = os.path.dirname(json_path)
        self.data = self._load_json()

    def _load_json(self):
        """
        Загружает JSON-отчёт из файла.

        :return: Словарь с данными отчёта
        """
        with open(self.json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def plot_static_diagnostics(self):
        """
        Создаёт и сохраняет PNG-график с анализом "силы" голосования по сценам.
        Визуализируются:
        - усреднённый сигнал изменений по трём признакам (OpenL3, YAMNet, RMS),
        - момент голосования за смену сцены.
        """
        detection_log = self.data["detection_log"]
        df = pd.DataFrame(detection_log)

        # Временные точки, где голосование одобрило сцену
        voted_times = [row["time"] for row in detection_log if row["voted"]]

        # Перевод признаков в числовую форму (1/0)
        audio_scores = df["audio_change"].astype(int)
        yamnet_scores = df["yamnet_change"].astype(int)
        rms_scores = df["rms_change"].astype(int)

        plt.figure(figsize=(12, 6))

        # === Верхний график: Усреднённый сигнал изменений ===
        plt.subplot(2, 1, 1)
        avg_signal = (audio_scores + yamnet_scores + rms_scores) / 3.0
        plt.plot(df["time"], avg_signal, label="Average Change Signal", color='blue')
        for t in voted_times:
            plt.axvline(x=t, color='red', alpha=0.4, linestyle='--')
        plt.title("Voting Signal Strength & Scene Change Points")
        plt.ylabel("Average Change")
        plt.grid(True)

        # === Нижний график: Линия голосования ===
        plt.subplot(2, 1, 2)
        plt.plot(df["time"], df["voted"].astype(int), label="Voted", color="black")
        for t in voted_times:
            plt.axvline(x=t, color='red', alpha=0.4, linestyle='--')
        plt.title("Scene Cut Voting Timeline")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Vote (1=Cut)")
        plt.grid(True)

        # Сохранение PNG-файла
        plt.tight_layout()
        filename = os.path.join(self.output_dir, f"{self.data['method']}_diagnostics.png")
        plt.savefig(filename)
        plt.close()
        print(f"✅ PNG-график сохранён: {filename}")

    def plot_interactive(self):
        """
        Создаёт и сохраняет интерактивную HTML-визуализацию по трём признакам
        и финальному голосованию модели.
        """
        df = pd.DataFrame(self.data["detection_log"])

        fig = go.Figure()

        # Добавление каждого признака как отдельной линии
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['audio_change'],
            mode='lines', name='Audio Change'
        ))
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['yamnet_change'],
            mode='lines', name='YAMNet Change'
        ))
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['rms_change'],
            mode='lines', name='RMS Change'
        ))
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['voted'],
            mode='lines', name='Voted Scene Cut',
            line=dict(width=2, color='black')
        ))

        # Настройка оформления графика
        fig.update_layout(
            title=f"Scene Detection Signals ({self.data['method']})",
            xaxis_title="Time (seconds)",
            yaxis_title="Binary Flags",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white",
            height=500
        )

        # Сохранение интерактивного графика
        filename = os.path.join(self.output_dir, f"{self.data['method']}_interactive.html")
        fig.write_html(filename)
        print(f"✅ HTML-график сохранён: {filename}")
