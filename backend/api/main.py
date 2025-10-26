"""FastAPI backend for Music AI App."""
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List, Dict
from sqlalchemy.orm import Session
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database.models import (
    init_db, get_db, GeneratedMusic, UserFeedback, ModelTrainingData
)
from backend.models.music_generator import MusicGenerator
from backend.models.music_utils import MusicFileGenerator

app = FastAPI(title="Music AI API", version="1.0.0")

# Initialize database
init_db()

# Initialize music generator
music_generator = MusicGenerator()
file_generator = MusicFileGenerator()


class MusicGenerationRequest(BaseModel):
    """Request model for music generation."""
    genre: str
    mood: str
    key_root: str
    mode: str
    theme: Optional[str] = None
    generations: int = 20


class FeedbackRequest(BaseModel):
    """Request model for user feedback."""
    music_id: int
    rating: int
    comment: Optional[str] = None


class MusicGenerationResponse(BaseModel):
    """Response model for music generation."""
    music_id: int
    file_path: str
    tempo: int
    metadata: Dict


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Music AI API", "version": "1.0.0"}


@app.post("/api/generate", response_model=MusicGenerationResponse)
async def generate_music(
    request: MusicGenerationRequest,
    db: Session = Depends(get_db)
):
    """Generate music based on parameters.
    
    Args:
        request: Music generation parameters
        db: Database session
        
    Returns:
        Generated music information
    """
    try:
        # Get feedback data for self-learning
        training_data = db.query(ModelTrainingData).filter(
            ModelTrainingData.genre == request.genre,
            ModelTrainingData.mood == request.mood
        ).all()
        
        feedback_data = [
            {
                'genre': td.genre,
                'mood': td.mood,
                'parameters': td.parameters,
                'avg_rating': td.avg_rating
            }
            for td in training_data
        ]
        
        # Update generator with feedback
        music_generator.learn_from_feedback(feedback_data)
        
        # Generate music
        music_data = music_generator.evolve_music(
            genre=request.genre,
            key_root=request.key_root,
            mode=request.mode,
            mood=request.mood,
            generations=request.generations
        )
        
        # Create MIDI file
        file_path = file_generator.create_midi_file(music_data)
        
        # Calculate duration
        duration = sum(dur for _, dur, _ in music_data['melody'])
        
        # Save to database
        generated_music = GeneratedMusic(
            genre=request.genre,
            mood=request.mood,
            theme=request.theme,
            key_root=request.key_root,
            mode=request.mode,
            tempo=music_data['tempo'],
            duration=duration,
            file_path=file_path
        )
        db.add(generated_music)
        db.commit()
        db.refresh(generated_music)
        
        return MusicGenerationResponse(
            music_id=generated_music.id,
            file_path=file_path,
            tempo=music_data['tempo'],
            metadata=music_data['metadata']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    db: Session = Depends(get_db)
):
    """Submit feedback for generated music.
    
    Args:
        request: Feedback data
        db: Database session
        
    Returns:
        Success message
    """
    try:
        # Validate rating
        if request.rating < 1 or request.rating > 5:
            raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")
        
        # Check if music exists
        music = db.query(GeneratedMusic).filter(
            GeneratedMusic.id == request.music_id
        ).first()
        
        if not music:
            raise HTTPException(status_code=404, detail="Music not found")
        
        # Save feedback
        feedback = UserFeedback(
            music_id=request.music_id,
            rating=request.rating,
            comment=request.comment
        )
        db.add(feedback)
        
        # Update or create training data
        training_data = db.query(ModelTrainingData).filter(
            ModelTrainingData.music_id == request.music_id
        ).first()
        
        if training_data:
            # Update existing training data
            old_count = training_data.feedback_count
            old_avg = training_data.avg_rating or 0
            new_count = old_count + 1
            new_avg = (old_avg * old_count + request.rating) / new_count
            
            training_data.avg_rating = new_avg
            training_data.feedback_count = new_count
        else:
            # Create new training data
            training_data = ModelTrainingData(
                music_id=request.music_id,
                genre=music.genre,
                mood=music.mood,
                parameters='{}',  # Could store more detailed parameters
                avg_rating=float(request.rating),
                feedback_count=1
            )
            db.add(training_data)
        
        db.commit()
        
        return {"message": "Feedback submitted successfully", "music_id": request.music_id}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/music/{music_id}")
async def get_music(music_id: int, db: Session = Depends(get_db)):
    """Get music information by ID.
    
    Args:
        music_id: Music ID
        db: Database session
        
    Returns:
        Music information
    """
    music = db.query(GeneratedMusic).filter(GeneratedMusic.id == music_id).first()
    
    if not music:
        raise HTTPException(status_code=404, detail="Music not found")
    
    return {
        "id": music.id,
        "genre": music.genre,
        "mood": music.mood,
        "theme": music.theme,
        "key_root": music.key_root,
        "mode": music.mode,
        "tempo": music.tempo,
        "duration": music.duration,
        "file_path": music.file_path,
        "created_at": music.created_at
    }


@app.get("/api/feedback/{music_id}")
async def get_feedback(music_id: int, db: Session = Depends(get_db)):
    """Get feedback for a specific music piece.
    
    Args:
        music_id: Music ID
        db: Database session
        
    Returns:
        List of feedback entries
    """
    feedback_list = db.query(UserFeedback).filter(
        UserFeedback.music_id == music_id
    ).all()
    
    return [
        {
            "id": fb.id,
            "rating": fb.rating,
            "comment": fb.comment,
            "created_at": fb.created_at
        }
        for fb in feedback_list
    ]


@app.get("/api/stats")
async def get_stats(db: Session = Depends(get_db)):
    """Get overall statistics.
    
    Args:
        db: Database session
        
    Returns:
        Statistics dictionary
    """
    total_music = db.query(GeneratedMusic).count()
    total_feedback = db.query(UserFeedback).count()
    
    avg_rating = db.query(UserFeedback).with_entities(
        UserFeedback.rating
    ).all()
    
    avg_rating_value = sum(r[0] for r in avg_rating) / len(avg_rating) if avg_rating else 0
    
    return {
        "total_music_generated": total_music,
        "total_feedback": total_feedback,
        "average_rating": round(avg_rating_value, 2)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
