import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import chat as chat_router
from app.api import documents as documents_router
from app.api import evaluation as evaluation_router
from app.api import auth as auth_router
from app.db.base import Base, engine
from app.core.config import settings

logger = logging.getLogger(__name__)

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
app.include_router(auth_router.router)


@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)
    _bootstrap_admin_user()


def _bootstrap_admin_user():
    """
    Bootstrap an admin user from environment variables on startup.

    Only runs when AUTH_MODE=local, BOOTSTRAP_ADMIN_PASSWORD is provided,
    and no admin user exists in the database.
    """
    if settings.AUTH_MODE != "local":
        return

    if not settings.BOOTSTRAP_ADMIN_PASSWORD:
        logger.info("BOOTSTRAP_ADMIN_PASSWORD not set; skipping admin bootstrap.")
        return

    from sqlalchemy.orm import Session
    from app.db.session import SessionLocal
    from app.security.models import User, UserRole
    from app.security.password import hash_password

    db: Session = SessionLocal()
    try:
        existing_admin = db.query(User).filter(User.role == UserRole.ADMIN).first()
        if existing_admin:
            logger.info("Admin user already exists; skipping bootstrap.")
            return

        admin_user = User(
            username=settings.BOOTSTRAP_ADMIN_USERNAME,
            email=settings.BOOTSTRAP_ADMIN_EMAIL,
            hashed_password=hash_password(settings.BOOTSTRAP_ADMIN_PASSWORD),
            role=UserRole.ADMIN,
            is_active=True,
            is_external=False,
        )
        db.add(admin_user)
        db.commit()
        logger.info(
            "Bootstrap admin user created: username=%s, email=%s",
            settings.BOOTSTRAP_ADMIN_USERNAME,
            settings.BOOTSTRAP_ADMIN_EMAIL,
        )
    except Exception:
        logger.exception("Failed to bootstrap admin user.")
        db.rollback()
    finally:
        db.close()


@app.get("/", tags=["Root"])
def root():
    return {"message": "AI Chatbot API"}


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy", "service": "backend"}