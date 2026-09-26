import re
from app.schemas.chat import ChatRequest, ChatResponse, MessageRole
from app.services.llm.base import LlmProvider


def _extract_ocr_body(content: str) -> str:
    """Return the OCR body embedded in a chunk payload.

    The prompt's ``[{n}] Source: <file>\n<content>`` block may carry a
    ``Source Type: Image\nOCR Text:\n...`` header (Phase 34A OCR chunks) or a
    ``Source image: ...`` / ``OCR:\n...`` header (image_knowledge chunks).
    For exact-text lookups, the mock text provider must surface the actual
    visible OCR line (e.g. ``Failed Reason: 902``) deterministically, not the
    header. If no OCR marker is found, fall back to the raw content.
    """
    if not content:
        return ""
    text = content.strip()
    # ``OCR Text:`` (Phase 34A) — slice after the marker.
    m = re.search(r"OCR Text:\s*", text, re.IGNORECASE)
    if m:
        return _strip_ocr_headers(text[m.end():])
    # ``OCR:`` on its own line (image_knowledge body).
    m2 = re.search(r"(?m)^\s*OCR:\s*$\s*(.*)$", text)
    if m2 and m2.group(1).strip():
        return _strip_ocr_headers(m2.group(1))
    return _strip_ocr_headers(text)


def _strip_ocr_headers(text: str) -> str:
    """Remove leading source/type/file/header lines so only visible content stays."""
    if not text:
        return ""
    out = []
    for line in text.splitlines():
        low = line.strip().lower()
        if low.startswith("source image:") or low.startswith("source type:") \
           or low.startswith("image type:") or low.startswith("file:") \
           or low.startswith("page:") or low.startswith("ocr text:") \
           or low == "ocr:":
            continue
        if line.strip():
            out.append(line.strip())
    return "\n".join(out)


class MockProvider(LlmProvider):
    """
    Mock LLM provider for testing and development.
    
    When the request message contains RAG context (has 'INFORMATION:' and 
    'USER QUESTION:' markers), this returns a mock RAG response that proves
    retrieval happened with proper citation handling.
    
    For non-RAG prompts, returns simple mock responses.
    """
    
    def __init__(self):
        self.provider_name = "mock"
    
    def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Process a chat request and return a mock response.
        """
        msg = request.message.strip()
        
        # Detect RAG prompt - contains context chunks with [1], [2], etc markers
        if "INFORMATION:" in msg and "USER QUESTION:" in msg:
            return self._mock_rag_response(msg)
        
        # Normal mock responses
        msg_lower = msg.lower()
        if msg_lower in ("hello", "hi", "hey"):
            text = "Hello! How can I help you today?"
        elif "?" in msg:
            text = "That's an interesting question. Here's a mock answer for now."
        else:
            text = f"You said: '{msg}'. This is a mock response."
        return ChatResponse(message=text, role=MessageRole.assistant)
    
    def _mock_rag_response(self, prompt: str) -> ChatResponse:
        """
        Generate a mock RAG response that proves retrieval happened.
        
        Parses the INFORMATION section to extract chunk content and citations.
        Returns an answer that references the retrieved chunks.
        
        This mock demonstrates what a real LLM would return when given
        properly formatted context with source citations.
        """
        # Extract the context/information section
        info_match = re.search(r'INFORMATION:(.*?)USER QUESTION:', prompt, re.DOTALL)
        query_match = re.search(r'USER QUESTION:(.*?)(?:GUIDELINES:|ANSWER:|$)', prompt, re.DOTALL)
        
        user_question = query_match.group(1).strip() if query_match else "the question"
        
        if info_match:
            info_section = info_match.group(1).strip()
            # Parse chunks from [1], [2], etc format
            chunk_pattern = r'\[(\d+)\]\s*Source:\s*(.*?)\n(.*?)(?=\[\d+\]|\Z)'
            chunks = re.findall(chunk_pattern, info_section, re.DOTALL)

            if chunks:
                # Build a mock RAG answer that references retrieved content
                answer_parts = []
                for chunk_num, source, content in chunks:
                    # Surface the most informative OCR excerpt so exact-text
                    # answers (e.g. an error code like "902") are preserved
                    # deterministically instead of depending on the external
                    # OpenRouter model deciding which visible token to echo.
                    body = _extract_ocr_body(content)
                    content_preview = body[:200] if body else (content.strip()[:200])
                    if content_preview:
                        answer_parts.append(
                            f"According to [{chunk_num}] ({source}), {content_preview}..."
                        )

                if answer_parts:
                    answer = f"Based on the retrieved information: " + " ".join(answer_parts[:3])
                    return ChatResponse(message=answer, role=MessageRole.assistant)
        
        # Fallback if parsing fails
        return ChatResponse(
            message=f"Mock RAG answer: Found information in the knowledge base to help answer your question about '{user_question}'.",
            role=MessageRole.assistant
        )