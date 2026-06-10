FROM python:3.10-slim

# Установка системных зависимостей для сборки, если они потребуются
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем только requirements.txt для кэширования слоев
COPY requirements.txt .

# Устанавливаем все зависимости в один шаг с указанием CPU-версии torch для совместимости
RUN pip install --no-cache-dir -r requirements.txt torch --extra-index-url https://download.pytorch.org/whl/cpu

# Копируем скрипт загрузки модели и запускаем его для кэширования
COPY download_models.py ./
ENV HF_HUB_DISABLE_PROGRESS_BARS=1
RUN python download_models.py

# Копируем остальной исходный код
COPY . .

# Создаем папку для хранения SQLite базы данных и Qdrant данных
RUN mkdir -p data

# Запуск бота
CMD ["python", "bot.py"]
