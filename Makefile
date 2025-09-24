SHELL := /bin/bash

.PHONY: start clean redis-clear

start:
	uv run python gradio_app.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -f assets/calendars/* || true

redis-clear:
	redis-cli -u $${REDIS_URL:-redis://localhost:6379} FLUSHALL
