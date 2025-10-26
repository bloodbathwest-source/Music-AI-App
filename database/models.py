"""
Database models for the Music AI App
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

Base = declarative_base()


class Song(Base):
    """Model for storing song information"""
    __tablename__ = 'songs'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(200), nullable=False)
    artist = Column(String(200), default="AI Generated")
    genre = Column(String(100))
    duration = Column(Float)  # in seconds
    file_path = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)
    lyrics = Column(Text)
    album_art_path = Column(String(500))
    key = Column(String(10))
    tempo = Column(Integer)
    is_ai_generated = Column(Boolean, default=True)


class Playlist(Base):
    """Model for storing playlist information"""
    __tablename__ = 'playlists'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    song_ids = Column(Text)  # Comma-separated song IDs


class UserPreference(Base):
    """Model for storing user preferences"""
    __tablename__ = 'user_preferences'
    
    id = Column(Integer, primary_key=True)
    theme = Column(String(20), default="dark")
    default_genre = Column(String(100))
    default_key = Column(String(10))
    default_tempo = Column(Integer, default=120)
    last_updated = Column(DateTime, default=datetime.utcnow)


# Database setup
def get_database_url():
    """Get database URL"""
    db_path = os.path.join(os.path.dirname(__file__), 'music_ai.db')
    return f'sqlite:///{db_path}'


def init_database():
    """Initialize the database"""
    engine = create_engine(get_database_url())
    Base.metadata.create_all(engine)
    return engine


def get_session():
    """Get database session"""
    engine = create_engine(get_database_url())
    Session = sessionmaker(bind=engine)
    return Session()
