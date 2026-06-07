from abc import ABC, abstractmethod
from app.schemas.chat import ChatRequest, ChatResponse

class LlmProvider(ABC):
    @abstractmethod
    def chat(self, request: ChatRequest) -> ChatResponse:
        """Process a chat request and return a response."""
        pass