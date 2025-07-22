# %%
import whisper
import json
from openai import OpenAI

# %%
class ASRProcessor:
    def __init__(self, model_name="base"):
        self.model = whisper.load_model(model_name)

    def transcribe(self, audio_path: str, language="ru", verbose=True, word_timestamps=True):
        result = self.model.transcribe(
            audio_path,
            language=language,
            verbose=verbose,
            word_timestamps=word_timestamps
        )
        return result

    @staticmethod
    def save_segments(segments, filepath="segments.json"):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)

    @staticmethod
    def get_short_segments(segments):
        return [
            {
                "start": round(seg["start"], 2),
                "end": round(seg["end"], 2),
                "text": seg["text"]
            }
            for seg in segments
        ]


# %%
class OpenAIThemer:
    def __init__(self, api_key: str, base_url: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def get_themes(self, segments_short):
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

        return response.choices[0].message.content

    @staticmethod
    def save_themes(themes_json, filepath="themes_with_ranges.json"):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(themes_json)


# %%
def main():
    VIDEO_PATH = "сваты_10мин.wav"

    # Распознавание речи
    asr = ASRProcessor()
    result = asr.transcribe(VIDEO_PATH)
    segments = result["segments"]
    asr.save_segments(segments, "segments2.json")
    segments_short = asr.get_short_segments(segments)

    # Группировка по темам
    themer = OpenAIThemer(
        api_key="",  # Добавь свой ключ
        base_url="https://api.proxyapi.ru/openai/v1"
    )
    themes_with_ranges = themer.get_themes(segments_short)
    themer.save_themes(themes_with_ranges, "themes_with_ranges2.json")

    print("Готово! Темы с общими диапазонами сохранены.")