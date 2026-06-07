import os
from app.schemas.chat import ChatRequest, ChatResponse, MessageRole
from app.services.llm.base import LlmProvider

class MockProvider(LlmProvider):
    def chat(self, request: ChatRequest) -> ChatResponse:
        msg = request.message.strip().lower()
        if msg in ("hello", "hi", "hey"):
            text = "Hello! How can I help you today?"
        elif "?" in request.message:
            text = "That's an interesting question. Here's a mock answer for now."
        else:
            text = f"You said: '{request.message}'. This is a mock response."
        return ChatResponse(message=text, role=MessageRole.assistant)