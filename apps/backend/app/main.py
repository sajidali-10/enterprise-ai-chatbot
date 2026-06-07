from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="AI Chatbot API",
    description="Enterprise AI Chatbot Backend",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
def root():
    return {"message": "AI Chatbot API"}


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy", "service": "backend"}