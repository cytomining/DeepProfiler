FROM python:3.12-slim

WORKDIR /app/deepprofiler
COPY . .
RUN pip install -e .

CMD ["python", "deepprofiler", "--help"]
