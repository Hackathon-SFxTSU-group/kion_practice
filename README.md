# KION_practice
## Установка библиотек
1. Успользуется CUDA 12.1 для работы torch и tenserflow с **GPU**

2. Необходимо установить ffmpeg https://www.gyan.dev/ffmpeg/builds/

1) Нужно скачать архив
2) Распаковать архив в удобное место ( Лучше на основной диск )
3) Прописать PATH к этой папке

3. Отдельно устанавливаем torch с GPU
```
pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 torchaudio==2.3.0+cu121 --index-url https://download.pytorch.org/whl/cu121

```
4. Установка всех остальных библиотек
```
pip install -r requirements.txt
```
5. Установка torchreid
```
pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
```