FROM python:3.12-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd -g 1000 app && useradd -u 1000 -g app -m -s /bin/sh app

WORKDIR /app/backend

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY --chown=app:app backend /app/backend
COPY --chown=app:app data/static /app/data/static
COPY --chown=app:app data/raw /app/data/raw

RUN mkdir -p /app/backend/staticfiles \
    && chown -R app:app /app/backend/staticfiles \
    && chmod +x /app/backend/deploy/entrypoint-web.sh


FROM base AS backend

USER app
EXPOSE 8000
ENTRYPOINT ["/app/backend/deploy/entrypoint-web.sh"]


FROM base AS ingestion

ARG TARGETARCH=arm64
ARG SUPERCRONIC_VERSION=v0.2.33

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        chromium \
        chromium-driver \
        tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && case "${TARGETARCH}" in \
        arm64) SUPERCRONIC_SHA1SUM=e0f0c06ebc5627e43b25475711e694450489ab00 ;; \
        amd64) SUPERCRONIC_SHA1SUM=71b0d58cc53f6bd72cf2f293e09e294b79c666d8 ;; \
        *) echo "Unsupported TARGETARCH: ${TARGETARCH}" >&2; exit 1 ;; \
       esac \
    && SUPERCRONIC_BIN="supercronic-linux-${TARGETARCH}" \
    && curl -fsSLO "https://github.com/aptible/supercronic/releases/download/${SUPERCRONIC_VERSION}/${SUPERCRONIC_BIN}" \
    && echo "${SUPERCRONIC_SHA1SUM}  ${SUPERCRONIC_BIN}" | sha1sum -c - \
    && chmod +x "${SUPERCRONIC_BIN}" \
    && mv "${SUPERCRONIC_BIN}" "/usr/local/bin/${SUPERCRONIC_BIN}" \
    && ln -s "/usr/local/bin/${SUPERCRONIC_BIN}" /usr/local/bin/supercronic

ENV CHROME_BINARY=/usr/bin/chromium \
    CHROMEDRIVER_PATH=/usr/bin/chromedriver \
    B3_INGEST_USE_SELENIUM=1 \
    TZ=America/Sao_Paulo

USER app
CMD ["supercronic", "/app/backend/deploy/crontab"]

