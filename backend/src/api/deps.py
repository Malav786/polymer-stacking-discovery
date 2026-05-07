from typing import Generator
from fastapi import Depends
from sqlalchemy.orm import Session
from src.db.schema import get_session_factory

SessionLocal = get_session_factory()

def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
