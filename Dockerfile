# Используйте официальный образ Python
FROM python:3.8

# Устанавливаем зависимости
RUN pip install fastapi uvicorn opencv-python-headless torch numpy

# Копируем все файлы из текущей директории в /app в контейнере
COPY . /app

# Устанавливаем рабочую директорию в /app
WORKDIR /app

# Запускаем ваш FastAPI приложение с помощью Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]