# Reproducible environment for the whole project.
#   docker build -t lime-repro .
#   docker run --rm -v "$PWD":/work lime-repro ./run_all.sh
FROM python:3.11.9-slim
WORKDIR /work
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["./run_all.sh"]
