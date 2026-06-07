from fastapi import APIRouter
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.llm import get_llm_provider

router = APIRouter(prefix="/api/chat", tags=["Chat"])

@router.post("", response_model=ChatResponse)
def post_chat(request: ChatRequest) -> ChatResponse:
    provider = get_llm_provider()
    return provider.chat(request)