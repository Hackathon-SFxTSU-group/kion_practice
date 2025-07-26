import logging
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util

# Настройка логгера
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TopicSegmenter:
    def __init__(self, model_name='ai-forever/FRIDA'):
        logger.info(f"Загрузка модели: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info("Модель успешно загружена.")

    def detect_topic_changes(self, segments, threshold=0.7, visualize=False):
        """
        segments: список dict {'text': str, 'start': float, 'end': float}
        threshold: float, порог изменения темы
        visualize: bool, если True — показать график изменений
        """
        logger.info(f"Обработка {len(segments)} сегментов текста...")
        texts = [seg['text'] for seg in segments]
        starts = [seg['start'] for seg in segments]

        logger.info("Вычисление эмбеддингов...")
        embeddings = self.model.encode(texts, convert_to_tensor=True)

        logger.info("Подсчёт семантических различий...")
        diffs = []
        for i in range(len(embeddings) - 1):
            sim = util.cos_sim(embeddings[i], embeddings[i + 1]).item()
            diff = 1 - sim
            diffs.append(diff)
            logger.debug(f"Сегмент {i} -> {i+1}: sim={sim:.4f}, diff={diff:.4f}")

        logger.info("Выявление смен тем...")
        cuts = [starts[i + 1] for i, d in enumerate(diffs) if d > threshold]
        logger.info(f"Обнаружено {len(cuts)} смен темы при пороге {threshold}.")

        if visualize:
            logger.info("Визуализация различий...")
            self._plot_differences(diffs, threshold, starts[1:])

        return cuts
    
    def detect_topic_changes_v2(self, segments, threshold=0.3, window_size=2, visualize=False):
        """
        Сравнивает кумулятивный контекст по обе стороны точки.
        :param segments: [{'text': str, 'start': float, 'end': float}, ...]
        :param threshold: float, порог 1 - similarity
        :param window_size: int, сколько сегментов брать до и после
        :param visualize: bool, отрисовать график
        """
        logger.info(f"Обработка {len(segments)} сегментов...")
        texts = [seg['text'] for seg in segments]
        starts = [seg['start'] for seg in segments]

        logger.info("Вычисление эмбеддингов...")
        embeddings = self.model.encode(texts, convert_to_tensor=True)

        logger.info("Сравнение кумулятивных контекстов...")
        diffs = []
        for i in range(window_size, len(embeddings) - window_size):
            left_context = embeddings[i - window_size:i].mean(dim=0)
            right_context = embeddings[i:i + window_size].mean(dim=0)

            sim = util.cos_sim(left_context, right_context).item()
            diff = 1 - sim
            diffs.append(diff)
            logger.debug(f"Точка {i}: sim={sim:.4f}, diff={diff:.4f}")

        # Смещения индексов из-за окна
        positions = starts[window_size:len(starts) - window_size]

        cuts = [positions[i] for i, d in enumerate(diffs) if d > threshold]
        logger.info(f"Обнаружено {len(cuts)} смен темы при пороге {threshold}.")

        if visualize:
            self._plot_differences(diffs, threshold, positions)

        return cuts

    def _plot_differences(self, diffs, threshold, positions):
        plt.figure(figsize=(12, 4))
        plt.plot(positions, diffs, marker='o', label='1 - similarity')
        plt.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
        for i, diff in enumerate(diffs):
            if diff > threshold:
                plt.axvline(x=positions[i], color='orange', linestyle=':', alpha=0.5)
        plt.xlabel('Start time of segment')
        plt.ylabel('1 - cosine similarity')
        plt.title('Semantic shift between adjacent segments')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
