"""Database models for Music AI App."""
from datetime import datetime

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class GeneratedMusic(Base):
    """Model for storing generated music pieces."""

    __tablename__ = 'generated_music'

    id = Column(Integer, primary_key=True, autoincrement=True)
    genre = Column(String(50), nullable=False)
    mood = Column(String(50), nullable=False)
    theme = Column(String(100))
    key_root = Column(String(10))
    mode = Column(String(20))
    tempo = Column(Integer)
    duration = Column(Float)
    file_path = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class UserFeedback(Base):
    """Model for storing user feedback on generated music."""

    __tablename__ = 'user_feedback'

    id = Column(Integer, primary_key=True, autoincrement=True)
    music_id = Column(Integer, nullable=False)
    rating = Column(Integer, nullable=False)  # 1-5 scale
    comment = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelTrainingData(Base):
    """Model for storing training data for self-learning."""

    __tablename__ = 'model_training_data'

    id = Column(Integer, primary_key=True, autoincrement=True)
    music_id = Column(Integer, nullable=False)
    genre = Column(String(50), nullable=False)
    mood = Column(String(50), nullable=False)
    parameters = Column(Text)  # JSON string of parameters
    avg_rating = Column(Float)
    feedback_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


# Database initialization
DATABASE_URL = "sqlite:///music_ai.db"
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
