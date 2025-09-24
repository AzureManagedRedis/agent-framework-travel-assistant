import json
import asyncio
from typing import Any, List, Mapping, Optional

from redis import Redis
from pydantic import BaseModel, Field
from typing_extensions import Self

from autogen_core._component_config import Component
from autogen_core.models import FunctionExecutionResultMessage, LLMMessage
from autogen_core.model_context._chat_completion_context import ChatCompletionContext, ChatCompletionContextState
from autogen_core.models import (
    AssistantMessage,
    SystemMessage,
    UserMessage,
    FunctionExecutionResultMessage,
)
from autogen_ext.models.openai import OpenAIChatCompletionClient


class RedisChatCompletionContextConfig(BaseModel):
    buffer_size: int = Field(default=6, description="Maximum number of messages to keep in buffer")
    redis_url: str = Field(default="redis://localhost:6379", description="Redis connection URL")
    redis_key_prefix: str = Field(default="chat_history", description="Redis key prefix for storing messages")
    user_id: str = Field(description="Unique identifier for this chat context")
    initial_messages: List[LLMMessage] | None = None
    summary_model: str = Field(default="gpt-4o-mini", description="OpenAI model used for summarization")


class RedisChatCompletionContext(ChatCompletionContext, Component[RedisChatCompletionContextConfig]):
    """A Redis-backed chat completion context with automatic conversation summarization.

    This implementation stores all messages persistently in Redis while providing
    efficient access to the most recent messages based on the buffer size. When the
    conversation grows beyond the buffer size, it automatically generates and maintains
    a rolling summary to preserve context.

    Args:
        buffer_size (int): The maximum number of messages to keep in the buffer.
        redis_url (str): Redis connection URL.
        redis_key_prefix (str): Prefix for Redis keys.
        user_id (str): Unique identifier for this chat context.
        initial_messages (List[LLMMessage] | None): The initial messages.
        summary_model (str): OpenAI model used for summarization.
    """

    component_config_schema = RedisChatCompletionContextConfig

    def __init__(
        self,
        buffer_size: int = 6,
        redis_url: str = "redis://localhost:6379",
        redis_key_prefix: str = "chat_history",
        user_id: str = "default",
        initial_messages: List[LLMMessage] | None = None,
        summary_model: str = "gpt-4.1-nano",
    ) -> None:
        if buffer_size <= 0:
            raise ValueError("buffer_size must be greater than 0.")
        
        self._buffer_size = buffer_size
        self._redis_url = redis_url
        self._redis_key_prefix = redis_key_prefix
        self._user_id = user_id
        self._summary_model = summary_model
        
        # Create Redis client
        self._redis_client = Redis.from_url(redis_url, decode_responses=True)
        
        # Redis keys for this context
        self._messages_key = f"{redis_key_prefix}:{user_id}:messages"
        self._summary_key = f"{redis_key_prefix}:{user_id}:summary"
        self._summary_upto_key = f"{redis_key_prefix}:{user_id}:summary_upto"
        
        # OpenAI client for summarization
        self._summary_client = OpenAIChatCompletionClient(model=summary_model)
        
        # Initialize Redis with initial messages if provided
        if initial_messages:
            self._initialize_redis_with_messages(initial_messages)

    def _initialize_redis_with_messages(self, messages: List[LLMMessage]) -> None:
        """Initialize Redis with the provided messages."""
        # Clear existing messages
        self._redis_client.delete(self._messages_key)
        
        # Add initial messages
        for message in messages:
            self._add_message_to_redis(message)

    def _add_message_to_redis(self, message: LLMMessage) -> None:
        """Add a message to Redis with proper ordering."""        
        # Serialize message with custom handling for complex objects
        serialized_data = self._serialize_message(message)
        # Add to Redis list (LPUSH for newest first)
        self._redis_client.lpush(self._messages_key, serialized_data)

    def _serialize_message(self, message: LLMMessage) -> str:
        """Serialize a message using Pydantic model_dump if available, fallback to __dict__."""
        return json.dumps(message.model_dump())


    def _deserialize_message(self, message_data: str) -> LLMMessage:
        """Deserialize a message from Redis storage."""
        data = json.loads(message_data)
        
        message_classes = {
            "AssistantMessage": AssistantMessage,
            "SystemMessage": SystemMessage,
            "UserMessage": UserMessage,
            "FunctionExecutionResultMessage": FunctionExecutionResultMessage,
        }
        
        # Get message type (works for both Pydantic model_dump and __dict__ formats)
        message_type = data.get("type")
        if not message_type:
            raise ValueError("Message data missing 'type' field")
        
        if message_type not in message_classes:
            raise ValueError(f"Unknown message type: {message_type}")
    
        return message_classes[message_type](**data)

    async def add_message(self, message: LLMMessage) -> None:
        """Add a message to the Redis-backed context."""
        self._add_message_to_redis(message)
        
        # Check if we should summarize
        if self._should_summarize():
            print(f"🧩 Summarization check passed for user {self._user_id}. Scheduling background update…")
            asyncio.create_task(self._update_summary())

    async def get_messages(self, n: Optional[int] = None) -> List[LLMMessage]:
        """Get at most `n` recent messages from Redis.
        
        Args:
            n (Optional[int]): Number of messages to retrieve. If None, uses buffer_size.
                              If -1, retrieves all messages.
        
        Returns:
            List[LLMMessage]: List of messages in chronological order (oldest to newest).
        """
        # Determine how many messages to retrieve
        if n is None:
            n = self._buffer_size
        
        # Get messages from Redis (LRANGE gets from newest to oldest)
        if n == -1:
            # Get all messages
            raw_messages = self._redis_client.lrange(self._messages_key, 0, -1)
        else:
            # Get the most recent n messages
            raw_messages = self._redis_client.lrange(self._messages_key, 0, n - 1)
        
        if not raw_messages:
            return []
        
        # Deserialize messages and reverse to get chronological order
        messages = []
        for raw_message in reversed(raw_messages):
            try:
                message = self._deserialize_message(raw_message)
                messages.append(message)
            except Exception as e:
                # Log error but continue processing other messages
                print(f"Error deserializing message: {e}")
                continue
        
        # Handle the case where first message is a function call result message
        if messages and isinstance(messages[0], FunctionExecutionResultMessage):
            # Remove the first message from the list
            messages = messages[1:]
        
        # Add summary as first message if available
        summary_text = self._redis_client.get(self._summary_key)
        if summary_text:
            summary_msg = SystemMessage(content=f"Previous conversation summary:\n{summary_text}")
            messages.insert(0, summary_msg)
        
        return messages
    
    @property
    def _messages(self) -> List[LLMMessage]:
        """Grab all messages from Redis in chronological order (oldest to newest)."""
        # This is a synchronous property, so we need to call the Redis operations directly
        # Get all messages from Redis (LRANGE gets from newest to oldest)
        raw_messages = self._redis_client.lrange(self._messages_key, 0, -1)
        
        if not raw_messages:
            return []
        
        # Deserialize messages and reverse to get chronological order
        messages = []
        for raw_message in reversed(raw_messages):
            try:
                message = self._deserialize_message(raw_message)
                messages.append(message)
            except Exception as e:
                # Log error but continue processing other messages
                print(f"Error deserializing message: {e}")
                continue
        
        # Handle the case where first message is a function call result message
        if messages and isinstance(messages[0], FunctionExecutionResultMessage):
            # Remove the first message from the list
            messages = messages[1:]
        
        return messages
    
    @_messages.setter
    def _messages(self, messages: List[LLMMessage]) -> None:
        """Set messages by clearing Redis and reinitializing with new messages.
        
        This setter is required because the base class methods like clear() and load_state()
        assign directly to self._messages.
        """
        # Clear existing messages in Redis
        self._redis_client.delete(self._messages_key)
        
        # Add all new messages to Redis
        if messages:
            self._initialize_redis_with_messages(messages)

    async def clear(self) -> None:
        """Clear all messages from the context and Redis."""
        self._redis_client.delete(self._messages_key)
        self._redis_client.delete(self._summary_key)
        self._redis_client.delete(self._summary_upto_key)


    def __del__(self):
        """Cleanup Redis connection when instance is destroyed."""
        self._redis_client.close()

    # ------------------------------
    # Incremental Summarization
    # ------------------------------
    
    def _should_summarize(self) -> bool:
        """Check if we should trigger summarization."""
        if not self._summary_client:
            return False
        
        # Summarize when we have more than buffer_size messages
        total_messages = self._redis_client.llen(self._messages_key)
        should = total_messages > self._buffer_size
        if should:
            print(f"🧪 Should summarize? yes — total={total_messages}, buffer={self._buffer_size}")
        else:
            print(f"🧪 Should summarize? no — total={total_messages}, buffer={self._buffer_size}")
        return should
    
    async def _update_summary(self) -> None:
        """Generate and store an updated conversation summary using only newly expired messages."""
        try:
            total = self._redis_client.llen(self._messages_key)
            buffer = self._buffer_size
            expired_count = max(0, total - buffer)
            last_summarized = int(self._redis_client.get(self._summary_upto_key) or 0)
            print(f"📏 Summary delta calc — total={total}, buffer={buffer}, expired={expired_count}, summarized_upto={last_summarized}")

            if expired_count <= last_summarized:
                print("ℹ️ No new expired messages to summarize.")
                return

            # Compute slice of newly expired messages (Redis newest-first indexing)
            start_idx = buffer + last_summarized
            end_idx = buffer + expired_count - 1
            print(f"📚 Fetching newly expired slice with LRANGE {start_idx}..{end_idx}")
            raw_slice = self._redis_client.lrange(self._messages_key, start_idx, end_idx)
            if not raw_slice:
                print("⚠️ Newly expired slice was empty; skipping.")
                self._redis_client.set(self._summary_upto_key, str(expired_count))
                return

            # Convert to readable turns in chronological order
            new_expired_turns = self._format_raw_messages_as_turns(list(reversed(raw_slice)))
            print(f"🧾 Newly expired usable turns: {len(new_expired_turns)}")
            if not new_expired_turns:
                print("⚠️ No user/assistant turns in newly expired slice; skipping.")
                self._redis_client.set(self._summary_upto_key, str(expired_count))
                return

            existing_summary = self._redis_client.get(self._summary_key)
            print(f"🧱 Existing summary present? {'yes' if existing_summary else 'no'}; folding in {len(new_expired_turns)} new turns…")

            new_summary = await self._generate_summary(existing_summary, new_expired_turns)
            if new_summary:
                self._redis_client.set(self._summary_key, new_summary)
                self._redis_client.set(self._summary_upto_key, str(expired_count))
                print(f"✅ Summary updated. summarized_upto -> {expired_count}")
            else:
                print("⚠️ LLM did not return a summary update; leaving previous summary intact.")
        except Exception as e:
            print(f"Error updating summary for user {self._user_id}: {e}")
    
    def _format_raw_messages_as_turns(self, raw_messages: List[str]) -> List[str]:
        """Format raw serialized messages into simple 'User:'/'Assistant:' turns."""
        turns: List[str] = []
        for raw_message in raw_messages:
            try:
                data = json.loads(raw_message)
                msg_type = data.get("type")
                content = (data.get("content") or "").strip()
                if not content:
                    continue
                if msg_type == "UserMessage":
                    turns.append(f"User: {content}")
                elif msg_type == "AssistantMessage":
                    turns.append(f"Assistant: {content}")
            except Exception:
                continue
        return turns
    
    async def _generate_summary(self, existing_summary: str | None, recent_messages: List[str]) -> str | None:
        """Generate a conversation summary using the OpenAI client."""
        if not recent_messages:
            return existing_summary
        
        # Build the prompt
        system_prompt = (
            "You are summarizing a travel planning conversation between a human and an AI travel concierge. "
            "Keep track of key details like destinations, timleines, preferences, decisions made, and open questions. "
            "Be concise but complete. Focus on key details and actionable information that will help the agent plan better. "
        )
        
        if existing_summary:
            user_prompt = (
                f"Previous summary:\n{existing_summary}\n\n"
                f"Recent conversation segments:\n" + "\n".join(recent_messages) + "\n\n"
                "Update the summary to include new information while keeping it concise."
            )
        else:
            user_prompt = (
                f"Conversation to summarize:\n" + "\n".join(recent_messages) + "\n\n"
                "Create a concise summary of this travel planning conversation."
            )
        
        try:
            # Use the autogen OpenAI client
            response = await self._summary_client.create([
                SystemMessage(content=system_prompt),
                UserMessage(content=user_prompt, source="user"),
            ])
            
            if response.content:
                return response.content.strip()
        except Exception as e:
            print(f"Error generating summary: {e}")
        
        return None
