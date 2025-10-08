"""Configuration management for the Travel Agent application.

Provides a typed `AppConfig` with environment-driven settings and basic
dependency checks used by the UI entrypoint.
"""
# import suppress_warnings  # Must be first to suppress warnings
import warnings
warnings.filterwarnings("ignore")
import os
from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    """Application configuration with validation."""
    
    # API Keys
    azure_openai_api_key: str = Field(..., env="AZURE_OPENAI_API_KEY", description="Azure OpenAI API key")
    tavily_api_key: str = Field(..., env="TAVILY_API_KEY", description="Tavily API key")

    azure_openai_base_url: str = Field(..., env="AZURE_OPENAI_BASE_URL", description="Azure OpenAI Base URL")
    azure_openai_api_version: str = Field(default="2024-06-01", env="AZURE_OPENAI_API_VERSION", description="Azure OpenAI API version")
    azure_openai_endpoint: str = Field(default="", env="AZURE_OPENAI_ENDPOINT", description="Azure OpenAI Endpoint")
    
    # Model Configuration
    travel_agent_model: str = Field(default="gpt-4.1", env="TRAVEL_AGENT_MODEL", description="OpenAI model name for the travel agent")
    mem0_model: str = Field(default="gpt-4.1-mini", env="MEM0_MODEL", description="OpenAI LLM name for the travel agent memory system")
    mem0_embedding_model: str = Field(default="text-embedding-3-small", env="MEM0_EMBEDDING_MODEL", description="OpenAI embedding model for Mem0 memory system")
    mem0_embedding_model_dims: int = Field(default=1536, env="MEM0_EMBEDDING_MODEL_DIMS", description="Embedding dimensions for OpenAI embedding model")

    # Other config
    max_tool_iterations: int = Field(default=8, env="MAX_TOOL_ITERATIONS", description="Maximum tool iterations")
    # Keep at least last 20 conversation steps (≈ 40 messages)
    max_chat_history_size: int = Field(default=40, env="MAX_CHAT_HISTORY_SIZE", description="Maximum chat history size (messages)")
    max_search_results: int = Field(default=5, env="MAX_SEARCH_RESULTS", description="Maximum search results from Tavily client")

    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL", description="Redis connection URL")
    
    # Mem0 mode selection
    mem0_cloud: bool = Field(default=False, env="MEM0_CLOUD", description="Use Mem0 Cloud when true; otherwise use local Mem0 with Redis vector store")

    # Server Configuration
    server_name: str = Field(default="0.0.0.0", env="SERVER_NAME", description="Server host")
    server_port: int = Field(default=7860, env="SERVER_PORT", description="Server port")
    share: bool = Field(default=False, env="SHARE", description="Enable public sharing")

    # Optional unless using Mem0 Cloud
    MEM0_API_KEY: Optional[str] = Field(default=None, env="MEM0_API_KEY", description="Mem0 API key (required if MEM0_CLOUD=true)")
    
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
    
    @model_validator(mode="after")
    def validate_mem0_requirements(self):  # type: ignore[override]
        """Ensure MEM0_API_KEY is present when using Mem0 Cloud."""
        if self.mem0_cloud and not (self.MEM0_API_KEY and self.MEM0_API_KEY.strip()):
            raise ValueError("MEM0_API_KEY is required when MEM0_CLOUD is true")
        return self



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
        
        client = OpenAI(api_key=config.azure_openai_api_key,
                        base_url = config.azure_openai_base_url)
        # Just test the client creation, not making an actual API call
        print("✅ OpenAI API key configured")
    except Exception as e:
        print(f"❌ OpenAI API error: {e}")
        return False
    
    print("✅ All dependencies validated")
    return True