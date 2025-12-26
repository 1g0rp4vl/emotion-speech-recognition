# Аннотация

## Постановка задачи

Задача состоит в том, чтобы научиться по небольшому отрывку речи человека распознавать его эмоцию. Это может быть нужно для создания автоматизированных колл-центров, чтобы LLM могла понять, что отвечать человеку, используя не только сухие текстовые данные, но и понимая, в каком состоянии находится человек.

## Формат входных и выходных данных

Входные данные: HTTP POST запрос, передающий айдиофайл.
Выходные данные: JSON, содержащий поле “emotion” – одно из 8: “neutral”, “happy”, “sad”, “angry”, “fearful”, “surprised”, “disgust”, а также словарь “probabilities”, содержащий вероятности, выданные моделью, для всех видов эмоций

## Метрики

Самое логичные метрики: accuracy – ожидается около 0.7 (в решениях с Kaggle есть accuracy 0.97, но они допускают ошибку не разделяя актеров на train и test, а просто случайно делят записи), f1-score (по классам отдельно и macro) – ожидаю, что-то около 0.6, это хороший результат по мнению интернет сообщества.

## Датасет

RAVDESS: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
Содержит по 60 записей для каждого из 24 актеров (12 мужчин и 12 женщин), каждая длительностью 3-5 секунд. (всего 1440 записей), содержащие предложения одной из 6 эмоций.

## Моделирование

### Бейзлайн

Например, можно использовать многоклассовую логистическую регрессию. Это хорошо интерпретируемая и не сложная в реализации модель

### Основная модель

Буду использовать WAV2VEC 2.0 (base) предобученную модель от Facebook, которая хорошо себя показала на задачах классификации эмоций в речи, c дополнительно обученной головой - полносвязным слоем с софтмаксом на 8 классов.

### Внедрение

Это будет REST API веб-сервис. Для работы нужен также preprocessing – обрезание файлов или паддинг и аугментация (для обучения, увеличение датасета за счет некоторых действий с файлом: добавление шума, сдвиг…)

# Структура проекта

```
.
├── .dvc
│   └── config
├── .dvcignore
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── README.md
├── conf
│   ├── callbacks
│   │   └── callbacks.yaml
│   ├── config.yaml
│   ├── inference
│   │   └── inference.yaml
│   ├── logger
│   │   └── mlflow.yaml
│   ├── module
│   │   └── module.yaml
│   ├── prepare_data
│   │   └── prepare_data.yaml
│   └── trainer
│       └── trainer.yaml
├── convert_with_docker.sh
├── data.dvc
├── emotion_speech_recognition
│   ├── __init__.py
│   ├── audio_classifier.py
│   ├── datamodule.py
│   ├── export_to_onnx.py
│   ├── inference.py
│   ├── inference_client.py
│   ├── inference_server.py
│   ├── model.py
│   ├── modern_audio_features.py
│   ├── prepare_dataset.py
│   └── train.py
├── model.onnx.dvc
├── model_checkpoints.dvc
├── plots
│   ├── train_loss.png
│   ├── val_acc.png
│   ├── val_f1.png
│   └── val_loss.png
├── pyproject.toml
├── register_model.py
├── trt_converter
│   └── Dockerfile
└── uv.lock
```

# Конфигурация

Проект использует [Hydra](https://hydra.cc/) для управления конфигурациями. Все конфигурационные файлы находятся в директории `conf/`.

*   `conf/config.yaml`: Основной файл конфигурации, собирающий остальные.
*   `conf/module/module.yaml`: Параметры модели и данных (размер батча, скорость обучения, количество воркеров, частота дискретизации, длительность аудио, количество классов).
*   `conf/trainer/trainer.yaml`: Параметры PyTorch Lightning Trainer (количество эпох, seed).
*   `conf/inference/inference.yaml`: Настройки инференса (пути к чекпоинтам, ONNX модели, хост и порт сервера).
*   `conf/logger/mlflow.yaml`: Настройки логгирования в MLflow (URI трекинга, имя эксперимента).
*   `conf/prepare_data/prepare_data.yaml`: Настройки подготовки датасета (URL, пути, использование DVC).

### Переопределение параметров через CLI

Любой параметр из конфигурационных файлов можно переопределить при запуске скрипта из командной строки.

Примеры:

*   Изменение размера батча и количества эпох при обучении:
    ```bash
    uv run ./emotion_speech_recognition/train.py module.batch_size=16 trainer.max_epochs=5
    ```

*   Изменение пути к файлу для клиента инференса:
    ```bash
    uv run ./emotion_speech_recognition/inference_client.py +file_path="path/to/audio.wav"
    ```

# Emotion Speech Recognition

## Setup

```
Инструкция далее будет описана для операционной системы Linux / MacOS с использованием bash.

# Клонирование репозитория
git clone https://github.com/1g0rp4vl/emotion-speech-recognition.git
cd emotion-speech-recognition

# Установка uv
# pip
pip install uv

# Создание и активация виртуального окружения, а также установка зависимостей
uv venv && source .venv/bin/activate && uv sync

# Установка хуков
pre-commit install

# Запуск проверок
pre-commit run --all-files

```

Прошу обратить внимание, что все команды далее должны быть выполнены из корневой директории проекта.

## Train

```
# Тренировка модели
# Здесь важно обратить внимание, что поскольку хранилище dvc у меня локальное, то чтобы пользоваться репозиторием, нужно скачать датасет самостоятельно, это можно сделать с помощью (по умолчанию, train.py уже скачивает датасет, но если нужно скачать его отдельно, то можно использовать prepare_dataset.py), но важно не забыть изменить параметр use_dvc на False в конфиге conf/prepare_data/prepare_data.yaml
uv run ./emotion_speech_recognition/prepare_dataset.py

Для логгирования в MLflow, нужно запустить MLflow сервер (если он не был запущен ранее):
mlflow server -p 8080

# После чего можно запускать тренировку модели (порт и URI сервера MLflow можно изменить в конфигах)
uv run ./emotion_speech_recognition/train.py
```

## Plots

В папке plots будут сохранены графики метрик обучения и валидации.

## Production preparation

Перед тем, как делать все шаги далее нужно произвести обучение и соответствующий чекпоинт модели должен быть положен в параметр cfg.inference.ckpt в конфиге conf/inference/inference.yaml

```
# Экспорт в ONNX
uv run ./emotion_speech_recognition/export_to_onnx.py
```

```
# Экспорт в TensorRT (требуется выполнить экспорт в ONNX перед этим шагом)
./convert_with_docker.sh
```

## Inference server

ONNX Runtime

```
# Запуск сервера инференса (этот inference_server.py использует ONNX Runtime)
uv run ./emotion_speech_recognition/inference_server.py
```

Запуск MLflow Serving

```
# Запуск MLflow Serving
Логгирует модель в MLflow, если это не было сделано ранее (перед этим нужно запустить mlflow server, например, командой ниже):
./register_model.py

# Запуск сервера инференса с помощью MLflow Serving
MLFLOW_TRACKING_URI={mlflow_server_tracking_uri} mlflow models serve -m models:/{your_model_name} -p 5000 --no-conda
```


## Inference client

Для запуска модели на аудиофайле с помощью клиента инференса, выполните команду:
Клиент инференса выполняет предобработку аудиофайла и отправляет запрос на сервер инференса, запущенный ранее.

```
# Запуск клиента инференса
uv run ./emotion_speech_recognition/inference_client.py +file_path="path/to/audio/file.wav"
```
