import os
from app.services.llm.base import LlmProvider
from app.services.llm.mock_provider import MockProvider
from app.services.llm.openai_compatible_provider import OpenAICompatibleProvider

def get_llm_provider() -> LlmProvider:
    provider_name = os.getenv("LLM_PROVIDER", "mock").lower().strip()
    if provider_name == "openai":
        return OpenAICompatibleProvider()
    return MockProvider()