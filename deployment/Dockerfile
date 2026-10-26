FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY deployment/app.py ./deployment/app.py
COPY models ./models

EXPOSE 5000

CMD ["python", "deployment/app.py"]