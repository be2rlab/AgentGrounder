from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from typing import Optional

from utils.config_loader import ModelConfig

def create_llm(config: ModelConfig, base_url: Optional[str] = None) -> ChatOllama:
    return ChatOllama(model=config.name, base_url=base_url or config.base_url, temperature=config.temperature, seed=config.seed, num_ctx=config.num_ctx) #, reasoning=False)

def create_embed_model(config: ModelConfig) -> OllamaEmbeddings:
    return OllamaEmbeddings(model=config.name, base_url=config.base_url)