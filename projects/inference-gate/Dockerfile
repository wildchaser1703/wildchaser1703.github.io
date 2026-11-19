FROM python:3.12-slim
WORKDIR /service
COPY pyproject.toml ./
COPY --chown=65532:65532 gate ./gate
# Optional managed-environment CA bundle stays a build secret, never an image layer.
RUN --mount=type=secret,id=build_ca \
    if [ -f /run/secrets/build_ca ]; then \
      PIP_CERT=/run/secrets/build_ca pip install --no-cache-dir .; \
    else pip install --no-cache-dir .; fi
USER 65532:65532
CMD ["uvicorn", "gate.api:app", "--host", "0.0.0.0", "--port", "8000"]
