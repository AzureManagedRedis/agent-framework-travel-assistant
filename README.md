
# 🌍 AI Travel Concierge (Agent Framework + Redis + Mem0)

A travel planning assistant with dual-layer memory: Redis-backed chat history and Mem0-powered long‑term memory. It provides time‑aware research via Tavily, uses OpenAI models for planning, and can export finalized itineraries to an ICS calendar file, all wrapped in a polished Gradio UI with per‑user contexts.

## 🧠 Key features
- **Dual-layer memory**: Short‑term chat history in Redis; long‑term preferences via Mem0 (OpenAI LLM + embeddings)
- **Per‑user isolation**: Separate memory contexts and chat history for each user
- **Time‑aware search**: Tavily integration for logistics (flights/hotels/transport) and destination research
- **Calendar export (ICS)**: Generate calendar files for itineraries and open the folder via UI
- **Gradio UI**: Chat, user tabs, live agent event logs, clear‑chat control
- **Configurable**: Pydantic settings via environment variables, `.env` support

## Recommended Demo Flow

Try the following query flow to test the agent!
1. Can you recommend things to do in Lisbon based on online opinion?
2. Can you find a flight leaving on the 14th of January 2026 and returning on the 16th?
3. How about a hotel for the stay? 
4. Okay, put together an itinerary and give me a calendar.
5. Click the Open Calendar button to add the routine to your schedule!

## 🧩 Architecture overview
- `gradio_app.py`: Launches the Gradio app, builds UI, wires event streaming, calendar open, and user switching
- `agent.py`: Implements `TravelAgent` using Agent Framework
  - Tools: `search_logistics`, `search_general`, `generate_calendar_ics`
  - Mem0 long‑term memory per user; Redis chat message store for short‑term context
  - Tavily search/extract for fresh web info; ICS generation via `ics`
- `config.py`: Pydantic settings and dependency checks
- `context/seed.json`: Seeded users and initial long‑term memory entries
- `assets/styles.css`: Custom theme and styling

## ✅ Prerequisites
- Python 3.11.x or 3.12.x (per `pyproject.toml` requires >=3.11,<3.13)
- Redis instance (local Docker, Redis Cloud, or Azure Managed Redis)
- API keys: OpenAI, Tavily, Mem0

## 🔐 Required environment variables
Provide via your environment or a `.env` file in the project root. Minimum required:
- `OPENAI_API_KEY` (must start with `sk-`; validated)
- `TAVILY_API_KEY`
- `MEM0_API_KEY`

Recommended/optional overrides (defaults shown):
- `TRAVEL_AGENT_MODEL` = `gpt-4o-mini`
- `MEM0_MODEL` = `gpt-5-nano`
- `MEM0_EMBEDDING_MODEL` = `text-embedding-3-small`
- `MEM0_EMBDDING_MODEL_DIMS` = `1536`
- `REDIS_URL` = `redis://localhost:6379`
- `MAX_CHAT_HISTORY_SIZE` = `6`
- `MAX_SEARCH_RESULTS` = `5`
- `SERVER_NAME` = `0.0.0.0`
- `SERVER_PORT` = `7860`
- `SHARE` = `false`

Example `.env` template:
```env
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=...
MEM0_API_KEY=...
REDIS_URL=redis://localhost:6379
TRAVEL_AGENT_MODEL=gpt-4o-mini
MEM0_MODEL=gpt-5-nano
MEM0_EMBEDDING_MODEL=text-embedding-3-small
MEM0_EMBDDING_MODEL_DIMS=1536
MAX_CHAT_HISTORY_SIZE=6
MAX_SEARCH_RESULTS=5
SERVER_NAME=0.0.0.0
SERVER_PORT=7860
SHARE=false
```

## 🗄️ Redis setup options
- Local (Docker):
```bash
docker run --name redis -p 6379:6379 -d redis:8.0.3
```
- Redis Cloud: create a free database and set `REDIS_URL`
- Azure Managed Redis: see Microsoft quickstart (entry tier works)

To clear all app data in Redis (chat history, summaries):
```bash
make redis-clear
```

## ▶️ Install & run (uv)
This project uses `uv` for environment and dependency management.
```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# From the project root
echo "Creating and syncing environment..."
uv sync

# Create .env from example (then edit values)
cp -n .env.example .env 2>/dev/null || true

# Start the app (opens browser)
uv run python gradio_app.py
# or
make start
```
The app launches at `http://localhost:7860`.

## 👤 Seed users and memory
- Users are defined in `context/seed.json` under `user_memories`
- On first run, each user's long‑term memory is seeded via Mem0
- The default selected user is the first key in `seed.json`
- Switch users via the tabs at the top of the UI

Example `context/seed.json`:
```json
{
  "user_memories": {
    "Alice": [ { "insight": "Prefers boutique hotels and walkable neighborhoods" } ],
    "Bob": [ { "insight": "Loves food tours and early morning flights" } ]
  }
}
```

## 💬 Using the app
- Ask for trip ideas, date‑bound logistics, or destination research
- The agent will call tools as needed:
  - `search_logistics(query, start_date?, end_date?)` for flights/hotels/transport
  - `search_general(query)` for activities, neighborhoods, dining, events
  - `generate_calendar_ics(...)` once your itinerary is finalized to produce an `.ics` file
- The right panel shows live agent events and tool logs
- Use “Clear Chat” to wipe the current user’s short‑term history from Redis

## 📅 Calendar export
- When an itinerary is finalized, the agent can export an `.ics` file
- Click “Open Calendar” to open the per‑user calendars folder in your OS file explorer
- Files are stored under `assets/calendars/<USER_ID>/`

## 🧰 Make targets
```bash
make start        # Run the app via uv
make clean        # Remove __pycache__ and calendars
make redis-clear  # FLUSHALL on $REDIS_URL (defaults to localhost)
```

## 🐛 Troubleshooting
- Missing API keys: app exits with a configuration error and hints for `.env`
- OpenAI key must start with `sk-` (validated in `config.py`)
- Redis connection errors: verify `REDIS_URL` and that Redis is reachable
- Mem0 errors when seeding: check `MEM0_API_KEY` and OpenAI settings
- Browser doesn’t open: navigate to `http://localhost:7860` manually

## 📦 Dependencies (selected)
- `agent-framework-project` (local path dependency in `pyproject.toml`)
- `redis`, `redisvl`, `pydantic`, `pydantic-settings`
- `openai`, `tavily-python`, `httpx`, `ics`, `gradio`

---

Built with Redis, Agent Framework, OpenAI, and Tavily. Enjoy planning!

