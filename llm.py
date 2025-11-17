"""
LLM Module - Unified interface for OpenAI and Ollama models
"""
import os
from typing import Optional, Literal

from langchain_openai import ChatOpenAI
from langchain_community.llms.ollama import Ollama


def get_llm(
    provider: Literal["openai", "ollama"] = "ollama",
    model: Optional[str] = None,
    temperature: float = 0.7,
    **kwargs
):
    """
    Get an LLM instance based on the provider.
    
    Args:
        provider: Either "openai" or "ollama"
        model: Model name (if None, uses defaults)
        temperature: Model temperature for generation
        **kwargs: Additional provider-specific arguments
    
    Returns:
        LLM instance (ChatOpenAI or Ollama)
    """
    if provider == "openai":
        # Default OpenAI model
        if model is None:
            model = "gpt-4o-mini"
        
        # Get API key from environment or kwargs
        api_key = kwargs.pop("api_key", os.getenv("OPENAI_API_KEY"))
        
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            **kwargs
        )
    
    elif provider == "ollama":
        # Default Ollama model
        if model is None:
            model = "deepseek-r1"
        
        # Get base URL from environment or use default
        base_url = kwargs.pop("base_url", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
        
        return Ollama(
            model=model,
            temperature=temperature,
            base_url=base_url,
            **kwargs
        )
    
    else:
        raise ValueError(f"Unsupported provider: {provider}. Choose 'openai' or 'ollama'.")


def get_llm_from_config(config: dict):
    """
    Get an LLM instance from a configuration dictionary.
    
    Args:
        config: Dictionary with 'provider', 'model', and other LLM settings
    
    Returns:
        LLM instance
    """
    provider = config.get("provider", "ollama")
    model = config.get("model", None)
    temperature = config.get("temperature", 0.7)
    
    # Extract additional kwargs
    kwargs = {k: v for k, v in config.items() if k not in ["provider", "model", "temperature"]}
    
    return get_llm(provider=provider, model=model, temperature=temperature, **kwargs)

