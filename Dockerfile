FROM python:3.11-slim

# --- sensible defaults ---
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    PORT=7860

WORKDIR /app

# Install uv directly from the official image (much faster than pip install)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files first for better Docker layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies using uv sync with frozen lockfile for reproducible builds
# --frozen ensures exact versions from lockfile, --no-dev skips dev dependencies
RUN uv sync --frozen --no-dev --no-install-project

# Copy the rest of your application code
COPY . .

# Install the project itself in the existing virtual environment
RUN uv pip install --no-deps .

# Expose Gradio's port and run using uv run for proper environment activation
EXPOSE 7860
CMD ["uv", "run", "python", "gradio_app.py"]
