# import suppress_warnings  # Must be first to suppress warnings
"""Travel assistant agent integrating Mem0 long-term memory, Redis-backed chat
history, Tavily web search, and simple calendar (.ics) generation.

This module defines the `TravelAgent` used by the Gradio UI:
- Per-user memory via Mem0 with Redis storage and retrieval
- Two scoped web search tools (logistics vs general) powered by Tavily
- A small sanitizer wrapper for OpenAI tool-call message ordering
- Streaming chat utilities that surface structured UI events for the side panel
"""
import warnings
warnings.filterwarnings("ignore")
import os
import sys
import asyncio
import json
import re
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, AsyncGenerator
import re
import hashlib
from queue import Queue

from tavily import TavilyClient
from ics import Calendar, Event, DisplayAlarm
from ics.grammar.parse import ContentLine

# Agent Framework imports
from agent_framework.openai import OpenAIChatClient, OpenAIResponsesClient
from agent_framework import ChatMessage, Role, TextContent
from agent_framework import FunctionCallContent, FunctionResultContent
from agent_framework._middleware import agent_middleware, AgentRunContext
from agent_framework._tools import ai_function
from agent_framework_mem0 import Mem0Provider
from agent_framework_redis._chat_message_store import RedisChatMessageStore
from agent_framework.exceptions import ServiceResponseException

from config import AppConfig



class NonBlockingMem0Provider(Mem0Provider):
    """Mem0 provider that does not block on messages_adding.

    Schedules the underlying add operation as a background task so agent
    streaming can complete without waiting for Mem0's LLM post-processing.
    """

    async def messages_adding(self, thread_id, new_messages) -> None:  # type: ignore[override]
        try:
            asyncio.create_task(self._messages_adding_bg(thread_id, new_messages))
        except Exception:
            # Best-effort: background scheduling failed, ignore to avoid blocking user flow
            pass

    async def _messages_adding_bg(self, thread_id, new_messages) -> None:
        try:
            await super().messages_adding(thread_id, new_messages)
        except Exception as e:
            # Log and swallow to prevent errors bubbling up from background task
            print(f"⚠️ Mem0 background add failed: {e}")


class _SanitizingChatMessageStore:
    """Wrapper for `RedisChatMessageStore` that enforces OpenAI tool-call ordering.

    Why this exists
    OpenAI expects any tool (role="tool") messages to be preceded by an
    assistant message that declared the corresponding tool call(s). Some SDKs and
    UIs may write tool results without the matching assistant call IDs, which can
    cause API errors. This wrapper sanitizes history to avoid that.

    Behavior
    - Drops leading tool-role messages.
    - Filters tool results (`FunctionResultContent`) whose `call_id` does not
      match any prior assistant `FunctionCallContent` in the conversation.
    """

    def __init__(self, inner_store: RedisChatMessageStore) -> None:
        self._inner = inner_store

    async def list_messages(self) -> List[ChatMessage]:
        raw = await self._inner.list_messages()
        return self._sanitize_messages(raw)

    async def add_messages(self, messages: List[ChatMessage]) -> None:
        await self._inner.add_messages(messages)

    async def serialize_state(self, **kwargs: Any) -> Any:
        return await self._inner.serialize_state(**kwargs)

    async def deserialize_state(self, serialized_store_state: Any, **kwargs: Any) -> None:
        await self._inner.deserialize_state(serialized_store_state, **kwargs)

    def _sanitize_messages(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        """Return messages pruned to a sequence valid for OpenAI tool-calls."""
        if not messages:
            return []

        # 1) Drop leading tool-role messages
        start_idx = 0
        for i, m in enumerate(messages):
            role_val = m.role.value if hasattr(m.role, "value") else str(m.role)
            if role_val != "tool":
                start_idx = i
                break
        else:
            return []

        msgs = messages[start_idx:]

        # 2) Track valid tool call ids from preceding assistant messages
        valid_call_ids: set[str] = set()
        sanitized: List[ChatMessage] = []
        for m in msgs:
            role_val = m.role.value if hasattr(m.role, "value") else str(m.role)
            if role_val == "assistant":
                for c in m.contents:
                    if isinstance(c, FunctionCallContent) and getattr(c, "call_id", None):
                        valid_call_ids.add(c.call_id)
                sanitized.append(m)
                continue

            if role_val == "tool":
                filtered_contents = [
                    c for c in m.contents
                    if isinstance(c, FunctionResultContent) and getattr(c, "call_id", None) in valid_call_ids
                ]
                if filtered_contents:
                    sanitized.append(
                        ChatMessage(
                            role=m.role,
                            contents=filtered_contents,
                            author_name=m.author_name,
                            message_id=m.message_id,
                            additional_properties=m.additional_properties,
                            raw_representation=m.raw_representation,
                        )
                    )
                continue

            sanitized.append(m)

        return sanitized

@dataclass
class UserCtx:
    """User-specific context containing Mem0 provider and agent instances.

    Attributes:
        mem0_provider: Mem0 provider instance for user-specific memory management
        agent: Main chat agent with tools and memory integration
    """
    mem0_provider: Mem0Provider
    agent: Any


class TravelAgent:
    """Travel planning agent with Mem0-powered personalized memory capabilities.

    This agent provides personalized travel planning services by maintaining
    separate Mem0 memory contexts for each user. Each user gets their own Mem0
    memory instance and supervisor agent that are cached for performance.

    Features:
    - Per-user memory isolation using Mem0 with Redis backend
    - Semantic memory search and retrieval via Mem0
    - Web search integration for current travel information
    - Chat history management with configurable buffer sizes
    - Automatic memory extraction and personalized recommendations

    Attributes:
    - config: Application configuration containing API keys and model settings
    - tavily_client: Web search client for travel information
    - chat_client: OpenAI client that creates the chat agent
    """

    def __init__(self, config: Optional[AppConfig] = None):
        """Initialize the TravelAgent with configuration and shared resources.
        
        Args:
            config: Application configuration. If None, loads default config.
        """
        if config is None:
            from config import get_config
            config = get_config()
        self.config = config

        # Set environment variables for SDK clients
        os.environ["OPENAI_API_KEY"] = config.openai_api_key
        os.environ["TAVILY_API_KEY"] = config.tavily_api_key

        # Initialize shared clients
        self.tavily_client = TavilyClient(api_key=config.tavily_api_key)
        self.chat_client = OpenAIResponsesClient(
            ai_model_id=config.travel_agent_model,
            api_key=config.openai_api_key,
        )

        # Initialize user context cache
        self._user_ctx_cache = {}
    
    async def initialize_seed_data(self) -> None:
        """Initialize seed users with their memories. Call this after creating the agent."""
        await self._init_seed_users()

    # ------------------------------
    # User Context Management
    # ------------------------------
    
    def _create_mem0_provider(self, user_id: str) -> Mem0Provider:
        """Create a Mem0 provider instance bound to a specific user/thread."""
        print(f"🧠 Creating Mem0 provider for user: {user_id}")
        # Uses MEM0_API_KEY from env by default
        return NonBlockingMem0Provider(
            user_id=user_id,
            thread_id=f"user:{user_id}",
            context_prompt=(
                "Relevant durable traveler facts and preferences (use to personalize replies):"
            ),
        )

    def _get_or_create_user_ctx(self, user_id: str) -> UserCtx:
        """Return a cached or new `UserCtx` with memory, agent, and history store.

        Creates and caches a complete user context including Mem0 memory,
        chat history management, and supervisor agent.

        Args:
            user_id: Unique identifier for the user

        Returns:
            UserCtx: Complete user context with Mem0 memory initialized
        """
        if user_ctx := self._user_ctx_cache.get(user_id):
            return user_ctx
        
        # Create Mem0 provider instance
        mem0_provider = self._create_mem0_provider(user_id)

        # Prepare Redis chat message store factory bound to user thread id
        def _store_factory() -> RedisChatMessageStore:
            base = RedisChatMessageStore(
                redis_url=self.config.redis_url,
                thread_id=f"{user_id}",
                key_prefix="chat_messages",
                max_messages=self.config.max_chat_history_size,
            )
            # Wrap with sanitizer so history is valid for OpenAI
            return _SanitizingChatMessageStore(base)  # type: ignore[return-value]

        # Create chat agent with tools, context provider, and Redis-backed history
        agent = self._create_agent(
            user_id=user_id,
            mem0_provider=mem0_provider,
            chat_message_store_factory=_store_factory,
        )

        # Provide minimal model_context adapter for UI clear()
        class _ModelContextAdapter:
            def __init__(self, redis_url: str, thread_id: str):
                base = RedisChatMessageStore(
                    redis_url=redis_url,
                    thread_id=thread_id,
                    key_prefix="chat_messages",
                    max_messages=self.config.max_chat_history_size,
                )
                self._store = _SanitizingChatMessageStore(base)

            async def clear(self) -> None:
                await self._store.clear()

        # Attach adapter to agent for backward compatibility with UI
        try:
            setattr(agent, "model_context", _ModelContextAdapter(self.config.redis_url, f"user:{user_id}"))
        except Exception:
            pass

        # Cache and return user context
        self._user_ctx_cache[user_id] = UserCtx(
            mem0_provider=mem0_provider,
            agent=agent,
        )
        return self._user_ctx_cache[user_id]

    def _load_seed_data(self) -> Dict[str, Any]:
        """Load seed data from `context/seed.json` adjacent to this module."""
        seed_file = Path(__file__).parent / "context" / "seed.json"
        with open(seed_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_all_user_ids(self) -> List[str]:
        """Return a unified list of user IDs from currently cached contexts."""
        return self._user_ctx_cache.keys()

    async def _init_seed_users(self) -> None:
        """Initialize seed users by inserting their memories into Mem0."""
        seed_data = self._load_seed_data()
        user_memories = seed_data.get("user_memories", {})
        
        for user_id, memories in user_memories.items():
            try:
                ctx = self._get_or_create_user_ctx(str(user_id))
                print(f"🌱 Seeding memory for user: {user_id}")
                for memo in memories:
                    try:
                        await ctx.mem0_provider.mem0_client.add(
                            messages=[{"role": "user", "content": memo.get("insight", "")}],
                            user_id=str(user_id),
                            run_id=ctx.mem0_provider.thread_id,
                            metadata={"source": "seed"},
                        )
                    except Exception as _e:
                        print(f"   ⚠️ Skipping seed entry due to error: {_e}")
                print(f"✅ Seeded {len(memories)} memories via Mem0 for user: {user_id}")
            except Exception as e:
                print(f"❌ Failed to seed memory for user {user_id}: {e}")
                continue

    def _create_agent(
        self,
        *,
        user_id: str,
        mem0_provider: Mem0Provider,
        chat_message_store_factory,
    ) -> Any:
        """Create the chat agent with tools, Mem0 context, and Redis-backed history."""
        print("🤖 Creating ChatAgent with tools...", flush=True)
        try:
            agent = self.chat_client.create_agent(
                name="agent",
                instructions=self._get_system_message(),
                tools=self._get_tools(),
                chat_message_store_factory=chat_message_store_factory,
                context_providers=mem0_provider,
            )
            print("✅ ChatAgent created successfully", flush=True)
            return agent
        except Exception as e:
            print(f"❌ Failed to create ChatAgent: {e}", flush=True)
            print(f"   Full traceback: {traceback.format_exc()}", flush=True)
            raise
    
    def _get_tools(self) -> List[Any]:
        """Return the list of tool-callable functions exposed to the agent."""
        tools: List[Any] = []
        tools.append(
            ai_function(
                self.search_logistics,
                name="search_logistics",
                description=(
                    "Time-aware logistics search ONLY: flights, hotels, and intercity/local transport. "
                    "Use for availability, schedules, prices, carriers/properties, or routes. "
                    "Arguments: query (required), start_date (optional, YYYY-MM-DD), end_date (optional, YYYY-MM-DD). "
                    "Always include dates when the user mentions a travel window; if ambiguous, ask for dates before booking guidance. "
                    "NEVER use this for activities, attractions, neighborhoods, or dining. "
                    "Results are restricted to reputable flight/hotel/transport sources; top URLs are deeply extracted."
                ),
            )
        )
        tools.append(
            ai_function(
                self.search_general,
                name="search_general",
                description=(
                    "Time-aware destination research: activities, attractions, neighborhoods, dining, events, local tips. "
                    "Use for up-to-date things to do, cultural context, and planning inspiration. "
                    "Arguments: query (required). "
                    "Scope searches to the relevant season/year when possible and prefer recent sources. "
                    "NEVER use this for flights, hotels, or transport logistics. "
                    "Example: 'things to do in Lisbon in June 2026'."
                ),
            )
        )
        tools.append(
            ai_function(
                self.generate_calendar_ics,
                name="generate_calendar_ics",
                description=(
                    "📅 Generate a downloadable calendar file (.ics) from a simple travel itinerary. "
                    "Use when you have a finalized schedule with dates and times. "
                    "Arguments: either events (array) OR a single event via title+date; plus trip_name (optional). "
                    "Single event fields: title, date (YYYY-MM-DD), start_time (optional), end_time (optional), location (optional), notes (optional). "
                    "Examples: date='2026-06-05', start_time='14:30', end_time='16:00'. "
                    "Returns file_path for user to open."
                ),
            )
        )
        print(f"🏁 Tool creation complete. {len(tools)} tools ready.", flush=True)
        return tools
    
    def _get_system_message(self) -> str:
        """Return the supervisor system message with roles, tool guidance, and style."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        return (
            f"You are an expert, time-aware, friendly Travel Concierge AI. Today is {today} (UTC). "
            "Assume your built in knowledge may be outdated; for anything time-sensitive, verify with tools.\n\n"
            "ROLE:\n"
            "- Discover destinations, plan itineraries, recommend accommodations, and organize logistics on behalf of the user.\n"
            "- Research current options, prices, availability, and on-the-ground activities using your tools.\n"
            "- Produce clear, actionable itineraries and booking guidance.\n"
            "- Regardless of your prior knowledge, always use search tools for current or future-state information.\n\n"
            "TOOL USAGE: You have access to the following helpful tools.\n"
            "- Use search_logistics ONLY for flights, hotels, or transport. Include start_date/end_date (YYYY-MM-DD) when known.\n"
            "- Use search_general for activities, attractions, neighborhoods, dining, events, or local tips. Include dates when relevant.\n"
            "- Use generate_calendar_ics when you have a finalized itinerary. Pass simple events array with title, date, optional times/location/notes.\n"
            "- Prefer recent sources (past 12–24 months) and pass explicit dates to tools whenever the user provides a time window.\n"
            "DISCOVERY:\n"
            "- If missing details, ask targeted questions (exact dates or window, origin/destination, budget, party size, interests,\n"
            "  lodging preferences, accessibility, loyalty programs).\n\n"
            "OUTPUT STYLE:\n"
            "- Be concise and prescriptive with your suggestions, followups, and recommendations.\n"
            "- Seek to be the best and friendliest travel agent possible. You are the expert after all.\n"
            "- Cite sources with titles and URLs for any tool-based claim.\n"
            "- Normalize to a single currency if prices appear; state assumptions.\n"
            "- For itineraries, list day-by-day with times and logistics.\n\n"
            "MEMORY:\n"
            "- Consider any appended important insights (long-term memory) from the user before answering and adapt to them.\n"
            "- Consider any relevant memories as helpful context but treat current session state as priority since it's current."
        )

    # -----------------
    # Tools
    # -----------------

    def _perform_search(
        self,
        query: str,
        search_type: str,
        include_domains: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Shared search logic with optional URL extraction.
        
        Args:
            query: Search query
            search_type: "logistics" or "general" for logging
            include_domains: Optional domain restrictions
            start_date: Optional start date for query enhancement
            end_date: Optional end date for query enhancement
            
        Returns:
            Dictionary with results and extractions
        """
        print(f"🔧 {search_type.upper()} SEARCH: {query}", flush=True)
        
        try:
            # Augment query with dates if provided
            enhanced_query = query
            if start_date:
                enhanced_query += f" from {start_date}"
            if end_date and end_date != start_date:
                enhanced_query += f" to {end_date}"
            
            search_kwargs = {
                "query": enhanced_query,
                "topic": "general",
                "search_depth": "advanced",
                "max_results": self.config.max_search_results,
            }
            
            if include_domains:
                search_kwargs["include_domains"] = include_domains


            results = self.tavily_client.search(**search_kwargs)
            
            if not results:
                print(f"⚠️ Empty results from Tavily", flush=True)
                return {"results": [], "extractions": []}

            # Filter results by score
            all_results = results.get("results", [])
            filtered_results = [r for r in all_results if r.get("score", 0) > 0.2]
            print(f"📊 Found {len(filtered_results)}/{len(all_results)} quality results", flush=True)
            
            results["results"] = filtered_results

            # Extract top 2 URLs for deeper context
            top_urls = [r.get("url") for r in filtered_results[:2] if r.get("url")]
            extractions: List[Dict[str, Any]] = []
            
            if top_urls:
                try:
                    extracted = self.tavily_client.extract(urls=top_urls)
                    if isinstance(extracted, dict) and extracted.get("results"):
                        extractions = extracted.get("results", [])
                    elif isinstance(extracted, list):
                        extractions = extracted
                    print(f"📄 Extracted {len(extractions)} content blocks", flush=True)
                except Exception as extract_e:
                    print(f"⚠️ URL extraction failed: {extract_e}", flush=True)

            results["extractions"] = extractions
            print(f"✅ {search_type.upper()} COMPLETE: {len(filtered_results)} results + {len(extractions)} extractions", flush=True)
            return results
            
        except Exception as e:
            error_msg = f"❌ {search_type.upper()} ERROR: {str(e)}"
            print(error_msg, flush=True)
            return {"error": error_msg, "results": [], "extractions": []}

    async def search_logistics(
        self,
        query: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """✈️🏨🚆 Logistics search: flights, hotels, and transport only.

        What it is for
        - Airfare and airline schedules, hotels/stays, and intercity transport (train/bus/ferry/car rental).

        How to use
        - Provide a concise query that includes the route or destination and constraints, e.g.:
          "JFK to LHR, nonstop preferred" or "hotels in Kyoto near Gion, mid-range" or "train Paris to Amsterdam".
        - Optionally include start_date and end_date as YYYY-MM-DD strings to guide availability windows.

        Behavior
        - Restricts sources to reputable flight/hotel/transport providers and aggregators.
        - Returns the strongest matches first and deeply extracts the top URLs for rich context.
        """
        include_domains = [
            # Flights / OTAs
            "expedia.com", "kayak.com", "travel.google.com",
            # Hotels / stays
            "booking.com", "hotels.com",
        ]
        
        return await asyncio.to_thread(
            self._perform_search,
            query,
            "logistics",
            include_domains,
            start_date,
            end_date,
        )

    async def search_general(
        self,
        query: str,
    ) -> Dict[str, Any]:
        """📍 General destination research: activities, attractions, neighborhoods, dining, events.

        What it is for
        - Up-to-date things to do, local highlights, neighborhoods to stay, dining ideas, and cultural context.

        How to use
        - Provide a destination/time-focused query, e.g., "things to do in Lisbon in June",
          "Barcelona food tours", "best neighborhoods to stay in Tokyo".

        Behavior
        - Runs an open web search (no logistics domains restriction) with raw content for context.
        """
        # print(f"🔧 {general} SEARCH: {query}", flush=True)
        return await asyncio.to_thread(
            self._perform_search,
            query,
            "general",
            None,
            None,
            None,
        )

    async def generate_calendar_ics(
        self,
        events: Optional[List[Dict[str, Any]]] = None,
        trip_name: Optional[str] = None,
        title: Optional[str] = None,
        date: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        location: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """📅 Generate a simple .ics calendar file from travel events (non-blocking)."""

        def _generate_calendar_ics_sync() -> Dict[str, Any]:
            print(f"🔧 CALENDAR GENERATION: Creating simple .ics file", flush=True)
            try:
                _events = events
                if not _events:
                    if title and date:
                        single_event: Dict[str, Any] = {
                            "title": title,
                            "date": date,
                        }
                        if start_time:
                            single_event["start_time"] = start_time
                        if end_time:
                            single_event["end_time"] = end_time
                        if location:
                            single_event["location"] = location
                        if notes:
                            single_event["notes"] = notes
                        _events = [single_event]
                    else:
                        return {"error": "No events provided", "file_path": None, "events_count": 0}

                calendar = Calendar()
                calendar.extra.append(ContentLine("X-WR-CALNAME", value=trip_name or "Travel Itinerary"))

                user_id = getattr(self, '_current_user_id', 'default')

                for event_data in _events:
                    event = self._create_simple_event(event_data, user_id)
                    if event:
                        calendar.events.add(event)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_name = re.sub(r'[^\w\-_]', '_', (trip_name or 'itinerary'))[:30]
                filename = f"{timestamp}_{safe_name}.ics"

                calendar_dir = Path(__file__).parent / "assets" / "calendars" / user_id
                calendar_dir.mkdir(parents=True, exist_ok=True)
                file_path = calendar_dir / filename

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(str(calendar))

                print(f"✅ CALENDAR COMPLETE: {len(_events)} events in {filename}", flush=True)

                return {
                    "file_path": str(file_path.absolute()),
                    "filename": filename,
                    "events_count": len(_events)
                }
            except Exception as e:
                error_msg = f"❌ CALENDAR ERROR: {str(e)}"
                print(error_msg, flush=True)
                return {"error": error_msg, "file_path": None, "events_count": 0}

        return await asyncio.to_thread(_generate_calendar_ics_sync)

    def _create_simple_event(self, event_data: Dict[str, Any], user_id: str) -> Optional[Event]:
        """Create a simple ICS event from basic event data.
        
        Args:
            event_data: Dict with title, date, optional start_time/end_time/location/notes
            user_id: User ID for UID generation
            
        Returns:
            Event object or None if creation failed
        """
        try:
            title = event_data.get("title", "").strip()
            date_str = event_data.get("date", "").strip()
            
            if not title or not date_str:
                print(f"   ⚠️ Skipping event: missing title or date", flush=True)
                return None
            
            event = Event()
            event.name = title
            
            # Parse date (YYYY-MM-DD format)
            event_date = datetime.fromisoformat(date_str).date()
            
            # Check if we have times
            start_time = event_data.get("start_time", "").strip()
            end_time = event_data.get("end_time", "").strip()
            
            if start_time:
                # Timed event
                start_hour, start_min = map(int, start_time.split(':'))
                event.begin = datetime.combine(event_date, datetime.min.time().replace(hour=start_hour, minute=start_min))
                
                if end_time:
                    end_hour, end_min = map(int, end_time.split(':'))
                    event.end = datetime.combine(event_date, datetime.min.time().replace(hour=end_hour, minute=end_min))
                else:
                    # Default 1 hour duration
                    event.end = event.begin + timedelta(hours=1)
                    
                # Add default reminder for timed events (30 minutes before)
                alarm = DisplayAlarm()
                alarm.trigger = event.begin - timedelta(minutes=30)
                alarm.description = f"Reminder: {title}"
                event.alarms.append(alarm)
                    
            else:
                # All-day event
                event.begin = event_date
                event.make_all_day()
            
            # Add optional fields
            if location := event_data.get("location", "").strip():
                event.location = location
                
            if notes := event_data.get("notes", "").strip():
                event.description = notes
            
            # Simple UID
            uid_source = f"{user_id}:{title}:{date_str}:{start_time}"
            uid_hash = hashlib.md5(uid_source.encode()).hexdigest()[:12]
            event.uid = f"{uid_hash}@travel-agent"
            
            return event
            
        except Exception as e:
            print(f"   ⚠️ Event creation failed: {e}", flush=True)
            return None

    # -----------------
    # Chat and Memory Interface  
    # -----------------
    
    async def stream_chat_turn_with_events(self, user_id: str, user_message: str) -> AsyncGenerator[tuple[str, dict | None], None]:
        """
        Yield (growing assistant reply, normalized event | None) pairs as the agent streams.

        Uses Agent Framework middleware to robustly detect and tag events on streaming updates.
        """
        # ------------------------------
        # Capture stdout/stderr prints as live UI events (start early)
        # Use a thread-safe Queue because tools may run in worker threads
        # ------------------------------
        log_queue: Queue[str | None] = Queue()

        class _StreamTee:
            def __init__(self, original_stream, queue: Queue[str | None]):
                self._original = original_stream
                self._queue = queue
                self._buffer = ""

            def write(self, data: str) -> int:
                # First, capture lines to the async UI queue so UI isn't blocked by original sink buffering
                self._buffer += data
                while "\n" in self._buffer:
                    line, self._buffer = self._buffer.split("\n", 1)
                    line = line.strip()
                    if line:
                        try:
                            self._queue.put_nowait(line)
                        except Exception:
                            pass
                # Then, mirror to the original stream best-effort
                try:
                    written = self._original.write(data)
                    try:
                        self._original.flush()
                    except Exception:
                        pass
                except Exception:
                    written = len(data)
                return written

            def flush(self) -> None:
                try:
                    self._original.flush()
                except Exception:
                    pass

        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = _StreamTee(original_stdout, log_queue)  # type: ignore[assignment]
        sys.stderr = _StreamTee(original_stderr, log_queue)  # type: ignore[assignment]

        ctx = self._get_or_create_user_ctx(user_id)

        # Store current user ID for calendar generation
        self._current_user_id = user_id

        def _html(icon: str, title: str, message: str) -> str:
            safe_icon = icon or ""
            safe_title = title or ""
            safe_msg = message or ""
            return (
                f"<div class='event-card'>"
                f"<div class='event-title'>{safe_icon} {safe_title}</div>"
                f"<div class='event-message'>{safe_msg}</div>"
                f"</div>"
            )

        def _event(event_type: str, icon: str, title: str, message: str) -> dict:
            return {
                "type": event_type,
                "html": _html(icon, title, message),
            }

        @agent_middleware
        async def _ui_events_middleware(context: AgentRunContext, next) -> None:
            """Agent middleware that annotates streaming updates with UI events in additional_properties.ui_events."""
            await next(context)

            base_stream = context.result

            async def _wrapped_stream():
                emitted_stream_start = False
                async for update in base_stream:  # type: ignore[misc]
                    ui_events: list[dict] = []

                    # Detect first token chunk
                    try:
                        text_chunks = [
                            it.text
                            for it in (getattr(update, "contents", []) or [])
                            if isinstance(it, TextContent) and getattr(it, "text", None)
                        ]
                    except Exception:
                        text_chunks = []
                    if text_chunks and not emitted_stream_start:
                        emitted_stream_start = True
                        ui_events.append({"type": "llm_token_stream_start"})

                    # Detect function calls
                    try:
                        fcalls = [it for it in (getattr(update, "contents", []) or []) if isinstance(it, FunctionCallContent)]
                    except Exception:
                        fcalls = []
                    if fcalls:
                        names: list[str] = []
                        for fc in fcalls:
                            try:
                                names.append(getattr(fc, "name", None) or "tool")
                            except Exception:
                                names.append("tool")
                        ui_events.append({"type": "tool_call", "tool_name": ", ".join(names) or "tool"})

                    # Detect function results and attempt to extract file paths
                    try:
                        fresults = [it for it in (getattr(update, "contents", []) or []) if isinstance(it, FunctionResultContent)]
                    except Exception:
                        fresults = []
                    if fresults:
                        file_path: str | None = None
                        try:
                            first_result = fresults[0]
                            result_content = getattr(first_result, "result", None)
                            if isinstance(result_content, dict):
                                file_path = result_content.get("file_path")
                            elif isinstance(result_content, list):
                                for item in result_content:
                                    if isinstance(item, dict) and item.get("file_path"):
                                        file_path = item.get("file_path")
                                        break
                            elif isinstance(result_content, str):
                                try:
                                    data = json.loads(result_content)
                                    if isinstance(data, dict):
                                        file_path = data.get("file_path")
                                except Exception:
                                    pass
                                if not file_path:
                                    try:
                                        m = re.search(r"(/[^\s\"]+\.ics)", result_content)
                                        if m:
                                            file_path = m.group(1)
                                    except Exception:
                                        pass
                        except Exception:
                            file_path = None
                        ui_events.append({
                            "type": "tool_result",
                            "tool_name": "generate_calendar_ics" if file_path else "Tool",
                            **({"file_path": file_path} if file_path else {}),
                        })

                    addl = getattr(update, "additional_properties", None) or {}
                    merged_events = list(addl.get("ui_events", []))
                    if ui_events:
                        merged_events.extend(ui_events)
                        try:
                            update = update.model_copy(update={"additional_properties": {**addl, "ui_events": merged_events}})
                        except Exception:
                            try:
                                setattr(update, "additional_properties", {**addl, "ui_events": merged_events})
                            except Exception:
                                pass
                    yield update

            context.result = _wrapped_stream()

        # Start streaming from Agent Framework with middleware that tags updates
        stream = ctx.agent.run_stream(messages=user_message, middleware=[_ui_events_middleware])

        buffer = ""
        yield buffer, _event("user_message", "", "User message sent", f'"{user_message}"')

        # Emit context retrieval/submission info similar to ChatAgent.run_stream
        async def _contents_to_text(contents: list[Any]) -> str:
            parts: list[str] = []
            for it in contents or []:
                try:
                    t = getattr(it, "text", None)
                    if t:
                        parts.append(t)
                except Exception:
                    continue
            return "\n".join(parts).strip()

        try:
            input_messages = ctx.agent._normalize_messages(user_message)  # use same normalization
            cp = getattr(ctx.agent, "context_providers", None)
            if cp is not None:
                provider_ctx = await cp.model_invoking(input_messages)
                if provider_ctx and getattr(provider_ctx, "contents", None):
                    ctx_text = await _contents_to_text(getattr(provider_ctx, "contents", []) or [])
                    if ctx_text:
                        snippet = ctx_text if len(ctx_text) <= 600 else (ctx_text[:600] + "…")
                        yield buffer, _event("context_retrieved", "🧠", "Context retrieved", snippet)
                        yield buffer, _event("context_submitted", "📎", "Context submitted", "Included retrieved context in system message")
        except Exception:
            pass

        try:
            yield buffer, _event("llm_response_start", "", "LLM thinking", f"LLM receives user input")

            # Queues to multiplex text updates and UI events
            text_queue: asyncio.Queue[str | None] = asyncio.Queue()
            event_queue: asyncio.Queue[dict | None] = asyncio.Queue()
            stream_error: ServiceResponseException | None = None

            async def _consume_stream() -> None:
                try:
                    async for update in stream:  # type: ignore[misc]
                        # Forward annotated UI events if present
                        try:
                            addl = getattr(update, "additional_properties", None) or {}
                            for e in list(addl.get("ui_events", [])):
                                try:
                                    await event_queue.put(dict(e))
                                except Exception:
                                    continue
                        except Exception:
                            pass

                        # Forward text chunk
                        try:
                            chunk_text = getattr(update, "text", "") or ""
                        except Exception:
                            chunk_text = ""
                        if chunk_text:
                            nonlocal buffer
                            buffer += chunk_text
                            await text_queue.put(buffer + ' <span class="thinking-animation">●●●</span>')
                except ServiceResponseException as sre:
                    nonlocal stream_error
                    stream_error = sre
                finally:
                    await text_queue.put(None)

            async def _consume_logs() -> None:
                try:
                    while True:
                        line = await asyncio.to_thread(log_queue.get)
                        if line is None:
                            break
                        try:
                            display_line = str(line)
                        except Exception:
                            display_line = ""
                        if display_line:
                            await event_queue.put({
                                "type": "tool_log",
                                "html": _html("", "", display_line),
                            })
                finally:
                    await event_queue.put(None)

            # Start consumers
            stream_task = asyncio.create_task(_consume_stream())
            logs_task = asyncio.create_task(_consume_logs())

            text_done = False
            events_done = False
            logs_shutdown_sent = False
            next_text_task: asyncio.Task | None = None
            next_event_task: asyncio.Task | None = None

            # Drain both queues, yielding as soon as items arrive
            while not (text_done and events_done):
                if not text_done and next_text_task is None:
                    next_text_task = asyncio.create_task(text_queue.get())
                if not events_done and next_event_task is None:
                    next_event_task = asyncio.create_task(event_queue.get())

                done, _ = await asyncio.wait(
                    {t for t in (next_text_task, next_event_task) if t is not None},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if next_text_task in done:
                    try:
                        item = next_text_task.result()
                    finally:
                        next_text_task = None
                    if item is None:
                        text_done = True
                        if not logs_shutdown_sent:
                            # End log capture and let logs consumer flush and terminate
                            try:
                                sys.stdout = original_stdout  # type: ignore[assignment]
                                sys.stderr = original_stderr  # type: ignore[assignment]
                            except Exception:
                                pass
                            try:
                                await asyncio.sleep(0)
                            except Exception:
                                pass
                            try:
                                log_queue.put(None)
                            except Exception:
                                pass
                            logs_shutdown_sent = True
                    else:
                        yield item, None

                if next_event_task in done:
                    try:
                        item = next_event_task.result()
                    finally:
                        next_event_task = None
                    if item is None:
                        events_done = True
                    else:
                        yield buffer, item

            # Ensure background tasks are finished
            try:
                await asyncio.gather(stream_task, logs_task)
            except Exception:
                pass

            # If the streaming failed due to tool-call ordering, run non-streaming fallback
            if stream_error is not None:
                try:
                    result = await ctx.agent.run(messages=user_message)
                    final_text = ""
                    file_path = None
                    for m in getattr(result, "messages", []) or []:
                        try:
                            if getattr(m, "role", None) == Role.ASSISTANT and getattr(m, "contents", None):
                                for c in m.contents:
                                    if isinstance(c, TextContent) and getattr(c, "text", None):
                                        final_text += c.text
                            if getattr(m, "role", None) == Role.TOOL and getattr(m, "contents", None):
                                for c in m.contents:
                                    if isinstance(c, FunctionResultContent):
                                        rc = getattr(c, "result", None)
                                        if isinstance(rc, dict) and rc.get("file_path"):
                                            file_path = rc.get("file_path")
                                        elif isinstance(rc, str):
                                            try:
                                                data = json.loads(rc)
                                                if isinstance(data, dict) and data.get("file_path"):
                                                    file_path = data.get("file_path")
                                            except Exception:
                                                pass
                        except Exception:
                            continue

                    if final_text:
                        buffer = final_text
                        yield buffer, None
                    if file_path:
                        yield buffer, {
                            "type": "tool_result",
                            "html": _html("📅", "generate_calendar_ics finished", "Tool execution completed"),
                            "tool_name": "generate_calendar_ics",
                            "file_path": file_path,
                        }
                except Exception:
                    # Re-raise the original to be handled by caller if fallback also fails
                    raise stream_error

        except ServiceResponseException as sre:
            # Handle OpenAI tool-call ordering error by falling back to non-streaming
            msg = str(sre)
            if "messages with role 'tool'" in msg and "preceding message with 'tool_calls'" in msg:
                try:
                    result = await ctx.agent.run(messages=user_message)
                    # Extract final assistant text
                    final_text = ""
                    file_path = None
                    for m in getattr(result, "messages", []) or []:
                        try:
                            # assistant text
                            if getattr(m, "role", None) == Role.ASSISTANT and getattr(m, "contents", None):
                                for c in m.contents:
                                    if isinstance(c, TextContent) and getattr(c, "text", None):
                                        final_text += c.text
                            # tool result
                            if getattr(m, "role", None) == Role.TOOL and getattr(m, "contents", None):
                                for c in m.contents:
                                    if isinstance(c, FunctionResultContent):
                                        rc = getattr(c, "result", None)
                                        if isinstance(rc, dict) and rc.get("file_path"):
                                            file_path = rc.get("file_path")
                                        elif isinstance(rc, str):
                                            try:
                                                data = json.loads(rc)
                                                if isinstance(data, dict) and data.get("file_path"):
                                                    file_path = data.get("file_path")
                                            except Exception:
                                                pass
                        except Exception:
                            continue

                    buffer = final_text or buffer
                    if buffer:
                        yield buffer, None
                    if file_path:
                        yield buffer, {
                            "type": "tool_result",
                            "html": _html("📅", "generate_calendar_ics finished", "Tool execution completed"),
                            "tool_name": "generate_calendar_ics",
                            "file_path": file_path,
                        }
                except Exception:
                    # If fallback also fails, re-raise original to surface in UI
                    raise sre
            else:
                # Unknown service error, re-raise to be handled by caller
                raise

        # Final yield to ensure thinking animation is removed
        if buffer:
            # Schedule background memory add and emit UI event
            try:
                asyncio.create_task(self.store_memory(user_id, user_message))
                yield buffer, _event("memory_add", "🧠", "Adding interaction to memory", "Queued for background storage")
            except Exception:
                pass

            yield buffer, _event("llm_response_end", "✅", f"LLM finished streaming", "")
            yield buffer, None


    # -----------------
    # Utility Methods
    # -----------------

    async def store_memory(self, user_id: str, user_message: str) -> None:
        """Store user message in memory via background task (non-blocking)."""
        async def _bg() -> None:
            try:
                ctx = self._get_or_create_user_ctx(user_id)
                await ctx.mem0_provider.mem0_client.add(
                    messages=[{"role": "user", "content": user_message}],
                    user_id=user_id,
                    run_id=ctx.mem0_provider.thread_id,
                    metadata={"source": "chat"},
                )
            except Exception as e:
                print(f"⚠️ Failed to store conversation memory for user {user_id}: {e}")

        try:
            asyncio.create_task(_bg())
        except Exception:
            # If task scheduling fails, ignore to keep UI responsive
            pass

    async def get_chat_history(self, user_id: str, n: Optional[int] = None) -> List[Dict[str, str]]:
        """Retrieve chat history for a user from Redis storage.
        
        Converts internal message objects to Gradio-compatible format,
        filtering for user and assistant messages with text content.
        
        Args:
            user_id: User identifier to get history for
            n: Number of messages to retrieve. If None, uses buffer_size.
               If -1, retrieves all messages.
            
        Returns:
            List[Dict[str, str]]: Message dictionaries with 'role' and 'content' 
                                keys suitable for Gradio chat interface
        """
        try:
            # Read directly from Redis chat store using stable per-user thread ID
            store = RedisChatMessageStore(
                redis_url=self.config.redis_url,
                thread_id=f"{user_id}",
                key_prefix="chat_messages",
                max_messages=self.config.max_chat_history_size,
            )
            all_messages: List[ChatMessage] = await store.list_messages()

            # Optionally limit number of messages
            if n is not None and n >= 0:
                all_messages = all_messages[-n:]

            gradio_messages: List[Dict[str, str]] = []
            for msg in all_messages:
                try:
                    role_val = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
                    # Extract primary text from contents
                    text = None
                    if getattr(msg, "text", None):
                        text = msg.text
                    elif getattr(msg, "contents", None):
                        for c in msg.contents:
                            if isinstance(c, TextContent) and getattr(c, "text", None):
                                text = c.text
                                break
                    if not text:
                        continue
                    if role_val in ("user", "assistant"):
                        gradio_messages.append({"role": role_val, "content": text})
                except Exception:
                    continue

            return gradio_messages
        except Exception as e:
            print(f"Error retrieving chat history for user {user_id}: {e}")
            return []

    def user_exists(self, user_id: str) -> bool:
        """Check if a user context exists in the cache.
        
        Args:
            user_id: User identifier to check
            
        Returns:
            bool: True if user context exists, False otherwise
        """
        return user_id in self._user_ctx_cache

    def reset_user_memory(self, user_id: str) -> None:
        """Reset a user's Mem0 memory by removing their cached context.
        
        This clears the user's cached context and forces recreation of
        a fresh Mem0 memory instance on next interaction.
        
        Args:
            user_id: User identifier whose memory should be reset
        """
        if user_id in self._user_ctx_cache:
            print(f"🗑️  Resetting Mem0 memory for user: {user_id}")
            self._user_ctx_cache.pop(user_id, None)
    