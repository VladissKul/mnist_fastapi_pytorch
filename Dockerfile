FROM python:3.8

RUN pip install fastapi uvicorn opencv-python-headless torch numpy

EXPOSE 1488

COPY . /app

WORKDIR /app

CMD ["python", "main.py"]