FROM python:3.11-slim

# --- sensible defaults ---
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    PORT=7860

WORKDIR /app

# 1) Install uv (fast resolver/installer)
RUN pip install --no-cache-dir --upgrade pip uv

# 2) If you have a lockfile that's up-to-date, copy it now for caching
#    (If your lock is stale, comment this out, rebuild, then re-enable after `uv lock`)
COPY pyproject.toml ./
# COPY uv.lock ./

# 3) Copy & install the local dependency FIRST (must contain its own pyproject.toml)
COPY agent-framework-project/python /app/agent-framework-project/python
RUN uv pip install --system --no-deps ./agent-framework-project/python

# 4) Copy the rest of your application
COPY . .

# 5) Install your app and remaining deps
#    If you use a lockfile, you can "uv pip sync" instead.
RUN uv pip install --system .

# 6) Expose Gradio’s port and run
EXPOSE 7860
CMD ["python", "gradio_app.py"]
