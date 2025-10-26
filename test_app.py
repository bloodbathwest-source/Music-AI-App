#!/usr/bin/env python3
"""
End-to-end test suite for Music AI App.
Tests all major components and their integration.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.models.music_generator import MusicGenerator
from backend.models.music_utils import MusicFileGenerator
from backend.database.models import (
    init_db, SessionLocal, GeneratedMusic, UserFeedback, ModelTrainingData
)


def test_music_generation():
    """Test music generation with different parameters."""
    print("\n=== Testing Music Generation ===")

    mg = MusicGenerator()

    # Test different genre/mood combinations
    test_cases = [
        ('pop', 'C', 'major', 'happy'),
        ('jazz', 'D', 'minor', 'dramatic'),
        ('classical', 'E', 'major', 'calm'),
        ('rock', 'G', 'minor', 'energetic'),
    ]

    for genre, key, mode, mood in test_cases:
        music_data = mg.evolve_music(genre, key, mode, mood, generations=3)
        assert len(music_data['melody']) == 32, f"Expected 32 notes, got {len(music_data['melody'])}"
        assert 'tempo' in music_data, "Missing tempo in music data"
        assert 'metadata' in music_data, "Missing metadata"
        print(f"✓ Generated {genre} music in {key} {mode} ({mood} mood)")

    print("✅ Music generation tests passed!")


def test_file_generation():
    """Test MIDI file and waveform generation."""
    print("\n=== Testing File Generation ===")

    mg = MusicGenerator()
    fg = MusicFileGenerator()

    # Generate music
    music_data = mg.evolve_music('pop', 'C', 'major', 'happy', generations=2)

    # Test MIDI creation
    midi_path = fg.create_midi_file(music_data)
    assert os.path.exists(midi_path), f"MIDI file not created: {midi_path}"
    print(f"✓ MIDI file created: {midi_path}")

    # Test waveform generation
    waveform, sample_rate = fg.generate_waveform_data(music_data)
    assert len(waveform) > 0, "Empty waveform"
    assert sample_rate == 22050, f"Expected sample rate 22050, got {sample_rate}"
    print(f"✓ Waveform generated: {len(waveform)} samples")

    # Test notation data
    notation = fg.get_music_notation_data(music_data)
    assert len(notation) == len(music_data['melody']), "Notation length mismatch"
    print(f"✓ Notation data generated: {len(notation)} notes")

    print("✅ File generation tests passed!")


def test_database_operations():
    """Test database operations."""
    print("\n=== Testing Database Operations ===")

    # Initialize database
    init_db()
    db = SessionLocal()

    # Generate music
    mg = MusicGenerator()
    fg = MusicFileGenerator()
    music_data = mg.evolve_music('pop', 'C', 'major', 'happy', generations=2)
    midi_path = fg.create_midi_file(music_data)

    # Test saving music
    generated = GeneratedMusic(
        genre='pop',
        mood='happy',
        theme='test',
        key_root='C',
        mode='major',
        tempo=music_data['tempo'],
        duration=sum(dur for _, dur, _ in music_data['melody']),
        file_path=midi_path
    )
    db.add(generated)
    db.commit()
    db.refresh(generated)
    print(f"✓ Music saved to database: ID {generated.id}")

    # Test saving feedback
    feedback = UserFeedback(
        music_id=generated.id,
        rating=5,
        comment='Test feedback'
    )
    db.add(feedback)
    db.commit()
    print(f"✓ Feedback saved to database")

    # Test saving training data
    training = ModelTrainingData(
        music_id=generated.id,
        genre='pop',
        mood='happy',
        parameters='{}',
        avg_rating=5.0,
        feedback_count=1
    )
    db.add(training)
    db.commit()
    print(f"✓ Training data saved to database")

    # Test querying
    total_music = db.query(GeneratedMusic).count()
    total_feedback = db.query(UserFeedback).count()
    assert total_music > 0, "No music in database"
    assert total_feedback > 0, "No feedback in database"
    print(f"✓ Database queries work: {total_music} music, {total_feedback} feedback")

    db.close()
    print("✅ Database operations tests passed!")


def test_self_learning():
    """Test self-learning mechanism."""
    print("\n=== Testing Self-Learning Mechanism ===")

    # Create initial generator
    mg1 = MusicGenerator()
    assert len(mg1.learned_patterns) == 0, "Expected no learned patterns initially"
    print("✓ Initial generator has no learned patterns")

    # Create feedback data
    feedback_data = [
        {
            'genre': 'pop',
            'mood': 'happy',
            'parameters': '{}',
            'avg_rating': 5.0
        },
        {
            'genre': 'jazz',
            'mood': 'dramatic',
            'parameters': '{}',
            'avg_rating': 4.5
        }
    ]

    # Create generator with feedback
    mg2 = MusicGenerator(feedback_data=feedback_data)
    assert len(mg2.learned_patterns) == 2, f"Expected 2 patterns, got {len(mg2.learned_patterns)}"
    print(f"✓ Generator learned {len(mg2.learned_patterns)} patterns from feedback")

    # Test learning from additional feedback
    new_feedback = [
        {
            'genre': 'pop',
            'mood': 'happy',
            'parameters': '{}',
            'avg_rating': 4.0
        }
    ]
    mg2.learn_from_feedback(new_feedback)
    print("✓ Generator updated with new feedback")

    print("✅ Self-learning mechanism tests passed!")


def test_music_quality():
    """Test music quality heuristics."""
    print("\n=== Testing Music Quality ===")

    mg = MusicGenerator()

    # Generate music
    music_data = mg.evolve_music('classical', 'C', 'major', 'calm', generations=10)

    # Check melody properties
    melody = music_data['melody']
    unique_notes = len(set(note for note, _, _ in melody))
    assert unique_notes > 5, f"Too few unique notes: {unique_notes}"
    print(f"✓ Melody has {unique_notes} unique notes")

    # Check harmony
    harmony = music_data.get('harmony', [])
    assert len(harmony) > 0, "No harmony generated"
    print(f"✓ Harmony has {len(harmony)} notes")

    # Check rhythm
    rhythm = music_data.get('rhythm', [])
    assert len(rhythm) > 0, "No rhythm pattern"
    print(f"✓ Rhythm pattern has {len(rhythm)} beats")

    # Check tempo is reasonable
    tempo = music_data['tempo']
    assert 60 <= tempo <= 200, f"Unusual tempo: {tempo}"
    print(f"✓ Tempo is reasonable: {tempo} BPM")

    print("✅ Music quality tests passed!")


def cleanup():
    """Clean up test files."""
    print("\n=== Cleaning Up ===")

    # Remove test MIDI files
    if os.path.exists('generated_music'):
        for file in os.listdir('generated_music'):
            if file.startswith('music_') and file.endswith('.mid'):
                os.remove(os.path.join('generated_music', file))
                print(f"✓ Removed {file}")

    print("✅ Cleanup complete!")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Music AI App - End-to-End Test Suite")
    print("=" * 60)

    try:
        test_music_generation()
        test_file_generation()
        test_database_operations()
        test_self_learning()
        test_music_quality()
        cleanup()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe Music AI App is ready to use!")
        print("Run: streamlit run app.py")
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
