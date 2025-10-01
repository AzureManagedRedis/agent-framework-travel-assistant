
# 🌍 AI Travel Concierge (Agent Framework + Redis + Mem0)

A travel planning assistant with dual-layer memory: Redis-backed chat history and Mem0-powered long‑term memory. It provides time‑aware research via Tavily, uses OpenAI models for planning, and can export finalized itineraries to an ICS calendar file, all wrapped in a polished Gradio UI with per‑user contexts.

## 🧠 Key features
- **Dual-layer memory**: Short‑term chat history in Redis; long‑term preferences via Mem0 (OpenAI LLM + embeddings)
- **Per‑user isolation**: Separate memory contexts and chat history for each user
- **Time‑aware search**: Tavily integration for logistics (flights/hotels/transport) and destination research
- **Calendar export (ICS)**: Generate calendar files for itineraries and open the folder via UI
- **Gradio UI**: Chat, user tabs, live agent event logs, clear‑chat control
- **Configurable**: Pydantic settings via environment variables, `.env` support

## 🧩 Architecture overview
- `gradio_app.py`: Launches the Gradio app, builds UI, wires event streaming, calendar open, and user switching
- `agent.py`: Implements `TravelAgent` using Agent Framework
  - Tools: `search_logistics`, `search_general`, `generate_calendar_ics`
  - Mem0 long‑term memory per user; Redis chat message store for short‑term context
  - Tavily search/extract for fresh web info; ICS generation via `ics`
- `config.py`: Pydantic settings and dependency checks
- `context/seed.json`: Seeded users and initial long‑term memory entries
- `assets/styles.css`: Custom theme and styling

A diagram of a software company:



## ✅ Prerequisites
- Python >=3.11 
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
- Azure Managed Redis: This is an easy way to get a fully managed service that runs natively on Azure. You will require an Azure subscription to get started. Achieve unmatched performance with costs as low as USD 12 per month. Alternative methods for deploying Redis are outlined below. See quickstart guide through Azure portal: https://learn.microsoft.com/en-us/azure/redis/quickstart-create-managed-redis

  Note: Enable access key-based authentication on your cache if you want to run the agent locally. Note the endpoint URL and the password; this will be used later.
  
- Local (Docker):
```bash
docker run --name redis -p 6379:6379 -d redis:8.0.3
```
- Redis Cloud: create a free database and set `REDIS_URL`

To clear all app data in Redis (chat history, summaries):
```bash
make redis-clear
```

## ▶️ Install & run (uv)
This project uses `uv` for environment and dependency management.
```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install brew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dotenv
brew install dotenv

# From the project root
echo "Creating and syncing environment..."
uv sync

# Create .env from example (then fill in values)
cp -n .env.example .env 2>/dev/null || true

# Start the app (opens browser)
dotenv run -- uv run gradio_app.py
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

## Recommended user flow

Try the following query flow to see the agent in action:

### Blank User (Mark) - No preferences (cold start case)
1. I like bears and the places where bears can be found. 
2. I think Christmas is quite a magical time of year and I enjoy visiting winter wonderland villages
3. What's a place to visit that I might like?
4. Can you give me a solid 3 day itinerary based on online recommendations? For what to do in (Pick one of the locations it mentions) 
5. Okay, can you recommend a flight and hotel? I live in (Pick a location) and I'm going to leave on 22nd December and will return 26th December. 
6. Give me the calendar. 

### Pre-filled User (Shreya) - User preferences known at startup
1. I'm planning to hit up Redis Released in London this year. Can you find more info about it?

## 🐛 Troubleshooting
- Missing API keys: app exits with a configuration error and hints for `.env`
- OpenAI key must start with `sk-` (validated in `config.py`)
- Redis connection errors: verify `REDIS_URL` and that Redis is reachable
- Mem0 errors when seeding: check `MEM0_API_KEY` and OpenAI settings
- Browser doesn’t open: navigate to `http://localhost:7860` manually

---

Built with Redis, Agent Framework, OpenAI, and Tavily. Enjoy!

