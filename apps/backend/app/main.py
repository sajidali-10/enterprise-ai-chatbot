from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import chat as chat_router
from app.api import documents as documents_router
from app.api import evaluation as evaluation_router
from app.db.base import Base, engine

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

app.include_router(chat_router.router)
app.include_router(documents_router.router)
app.include_router(evaluation_router.router)


@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)


@app.get("/", tags=["Root"])
def root():
    return {"message": "AI Chatbot API"}


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy", "service": "backend"}