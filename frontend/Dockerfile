FROM python:3.10-slim
WORKDIR /app
COPY ui-requirements.txt .
RUN pip install --no-cache-dir -r ui-requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
