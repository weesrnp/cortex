# Estágio 1: dependências num venv (funciona com qualquer uid, ao contrário do pip --user)
FROM python:3.12-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt

# Estágio 2: execução
FROM python:3.12-slim-bookworm AS runtime

WORKDIR /app

# libpq para o psycopg e curl para o healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends libpq5 curl \
    && rm -rf /var/lib/apt/lists/*

# Usuário não-root. No Linux o compose roda com o uid do dono do cofre (CORTEX_UID), para poder gravar notas nele.
RUN useradd -u 1000 -m -s /bin/bash cortex

COPY --from=builder /opt/venv /opt/venv
COPY src/ /app/src/
COPY tests/ /app/tests/

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER cortex

EXPOSE 8900

CMD ["python", "-m", "src.main"]
