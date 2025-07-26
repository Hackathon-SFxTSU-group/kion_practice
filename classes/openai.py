import json
import os
import time
from openai import OpenAI


class OpenAIThemer:
    """
    Класс для тематической группировки распознанных речевых сегментов (ASR)
    с использованием LLM-модели OpenAI (например, GPT-4o).
    """

    def __init__(self, api_key: str, base_url: str):
        """
        Инициализация клиента OpenAI.

        :param api_key: API-ключ для доступа к OpenAI
        :param base_url: Базовый URL (например, локальный сервер или https://api.openai.com/v1)
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def get_themes(self, segments_short, audio_path: str):
        """
        Отправляет сегменты речи в LLM для смысловой группировки по темам.

        :param segments_short: Список речевых сегментов в формате [{"start": ..., "end": ..., "text": ...}, ...]
        :param audio_path: Путь к аудиофайлу (используется для имени отчета)
        :return: Словарь с темами: {"themes": [{"theme": ..., "time": [start, end]}, ...]}
        """
        # Формируем промпт для модели
        prompt = (
            "Вот список сегментов речи с временными метками:\n\n"
            f"{json.dumps(segments_short, ensure_ascii=False, indent=2)}\n\n"
            "Проанализируй их и сгруппируй в смысловые блоки. "
            "Для каждого блока укажи:\n"
            "- краткое название темы,\n"
            "- один общий временной диапазон, который охватывает все сегменты этой темы "
            "(в формате: [start, end], где start — начало первого сегмента, end — конец последнего).\n"
            "Верни ответ строго в JSON в формате:\n"
            "{ \"themes\": [ { \"theme\": \"Название темы\", \"time\": [start, end] }, ... ] }"
        )

        # Замер времени
        t0 = time.time()
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "Ты — помощник для группировки транскриптов по темам."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        duration = round(time.time() - t0, 2)

        # Получаем текст из ответа модели
        content = response.choices[0].message.content

        # Пытаемся распарсить JSON
        try:
            themes_data = json.loads(content)
        except json.JSONDecodeError:
            print("❌ Ошибка разбора JSON из ответа модели.")
            themes_data = {"themes": []}

        # Сохраняем результат в отчёт
        self.save_themes_report(themes_data, audio_path, duration)

        return themes_data

    def save_themes_report(self, themes_data, audio_path, duration_sec):
        """
        Сохраняет результат группировки в JSON-файл.

        :param themes_data: Словарь {"themes": [...]}
        :param audio_path: Путь к аудиофайлу (используется для имени файла)
        :param duration_sec: Время обработки в секундах
        """
        os.makedirs("reports", exist_ok=True)
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = os.path.join("reports", f"{base_name}_themes_report.json")

        report = {
            "audio_path": audio_path,
            "num_themes": len(themes_data.get("themes", [])),
            "processing_time_sec": duration_sec,
            "themes": themes_data.get("themes", [])
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"🧠 Тематический отчёт сохранён: {output_path}")
