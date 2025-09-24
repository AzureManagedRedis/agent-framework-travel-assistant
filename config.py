"""Configuration management for the Travel Agent application.

Provides a typed `AppConfig` with environment-driven settings and basic
dependency checks used by the UI entrypoint.
"""
# import suppress_warnings  # Must be first to suppress warnings
import warnings
warnings.filterwarnings("ignore")
import os
from typing import Optional
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    """Application configuration with validation."""
    
    # API Keys
    openai_api_key: str = Field(..., env="OPENAI_API_KEY", description="OpenAI API key")
    tavily_api_key: str = Field(..., env="TAVILY_API_KEY", description="Tavily API key")

    # Model Configuration
    travel_agent_model: str = Field(default="gpt-4.1", env="TRAVEL_AGENT_MODEL", description="OpenAI model name for the travel agent")
    mem0_model: str = Field(default="gpt-4.1-mini", env="MEM0_MODEL", description="OpenAI LLM name for the travel agent memory system")
    mem0_embedding_model: str = Field(default="text-embedding-3-small", env="MEM0_EMBEDDING_MODEL", description="OpenAI embedding model for Mem0 memory system")
    mem0_embedding_model_dims: int = Field(default=1536, env="MEM0_EMBDDING_MODEL_DIMS", description="Embedding dimensions for OpenAI embedding model")

    # Other config
    max_tool_iterations: int = Field(default=8, env="MAX_TOOL_ITERATIONS", description="Maximum tool iterations")
    max_chat_history_size: int = Field(default=6, env="MAX_CHAT_HISTORY_SIZE", description="Maximum chat history size")
    max_search_results: int = Field(default=5, env="MAX_SEARCH_RESULTS", description="Maximum search results from Tavily client")

    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL", description="Redis connection URL")
    
    # Server Configuration
    server_name: str = Field(default="0.0.0.0", env="SERVER_NAME", description="Server host")
    server_port: int = Field(default=7860, env="SERVER_PORT", description="Server port")
    share: bool = Field(default=False, env="SHARE", description="Enable public sharing")

    MEM0_API_KEY: str = Field(..., env="MEM0_API_KEY", description="Mem0 API key")
    
    class Config:
        """Pydantic config."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra environment variables
    
    @field_validator("openai_api_key")
    @classmethod
    def validate_openai_key(cls, v):
        """Validate OpenAI API key format."""
        if not v.startswith("sk-"):
            raise ValueError("OpenAI API key must start with 'sk-'")
        return v
    



def get_config() -> AppConfig:
    """Get application configuration with proper error handling."""
    try:
        return AppConfig()
    except Exception as e:
        print(f"❌ Configuration Error: {e}")
        print("\n📝 Please check your environment variables or create a .env file with:")
        print("OPENAI_API_KEY=sk-your-key-here")
        print("TAVILY_API_KEY=your-key-here")
        raise SystemExit(1)


def validate_dependencies() -> bool:
    """Validate that required services are available."""
    from openai import OpenAI
    
    config = get_config()
    
    # Test OpenAI API
    try:
        client = OpenAI(api_key=config.openai_api_key)
        # Just test the client creation, not making an actual API call
        print("✅ OpenAI API key configured")
    except Exception as e:
        print(f"❌ OpenAI API error: {e}")
        return False
    
    print("✅ All dependencies validated")
    return True
