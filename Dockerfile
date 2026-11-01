# Runs the service from a built wheel, as a non-root user, with no dev dependencies.
# TLS and network policy are the deployment's job: see docs/operations/security.md.
FROM python:3.12-slim AS build
WORKDIR /src
RUN pip install --no-cache-dir uv==0.5.11
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN uv build --wheel --out-dir /dist

FROM python:3.12-slim
RUN useradd --create-home --uid 10001 gn
COPY --from=build /dist /dist
RUN pip install --no-cache-dir "$(ls /dist/*.whl)[service]" && rm -rf /dist
# A named volume takes its ownership from this directory, so the audit log stays writable.
RUN mkdir -p /home/gn/data && chown gn /home/gn/data
USER gn
WORKDIR /home/gn
ENV GN_AUDIT_LOG=/home/gn/data/audit.jsonl
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"
CMD ["uvicorn", "--factory", "greeks_neutralizer.service.app:factory", "--host", "0.0.0.0", "--port", "8000"]
