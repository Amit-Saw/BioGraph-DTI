# CPU image.  Build:  docker build -t dti-app .          (add --build-arg WITH_DOCKING=1 for the Dock page)
# Run:    docker compose up -d                           (mounts checkpoints/, data/ and cache/)
FROM python:3.11-slim
ARG WITH_DOCKING=0
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 DTI_HOST=0.0.0.0 PORT=8000
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && if [ "$WITH_DOCKING" = "1" ]; then apt-get install -y --no-install-recommends openbabel; fi \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt requirements-docking.txt ./
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt \
    && if [ "$WITH_DOCKING" = "1" ]; then pip install -r requirements-docking.txt; fi
COPY . .
EXPOSE 8000
CMD ["gunicorn", "-c", "gunicorn.conf.py", "wsgi:app"]
