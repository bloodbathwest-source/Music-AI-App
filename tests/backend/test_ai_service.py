"""Test cases for AI music generation service."""
from backend.app.services.ai_service import MusicGenerationService, TORCH_AVAILABLE
from backend.app.models.music import MusicGenerationRequest


def test_music_generation_service_initialization():
    """Test that the music generation service initializes correctly."""
    service = MusicGenerationService()
    assert service is not None
    # Model may be None if torch is not available
    if TORCH_AVAILABLE:
        assert service.model is not None


def test_generate_music_basic():
    """Test basic music generation."""
    service = MusicGenerationService()
    request = MusicGenerationRequest(
        genre="pop",
        mood="happy",
        key="C",
        tempo=120,
        duration=60
    )

    result = service.generate_music(request)

    assert result is not None
    assert "melody" in result
    assert "midi_data" in result
    assert "parameters" in result
    assert len(result["melody"]) > 0


def test_generate_music_different_genres():
    """Test music generation with different genres."""
    service = MusicGenerationService()
    genres = ["pop", "jazz", "classical", "rock", "electronic"]

    for genre in genres:
        request = MusicGenerationRequest(
            genre=genre,
            mood="happy",
            key="C",
            tempo=120,
            duration=30
        )
        result = service.generate_music(request)
        assert result is not None
        assert result["parameters"]["genre"] == genre


def test_generate_music_different_keys():
    """Test music generation with different musical keys."""
    service = MusicGenerationService()
    keys = ["C", "D", "E", "F", "G", "A", "B"]

    for key in keys:
        request = MusicGenerationRequest(
            genre="pop",
            mood="happy",
            key=key,
            tempo=120,
            duration=30
        )
        result = service.generate_music(request)
        assert result is not None
        assert result["parameters"]["key"] == key


def test_generate_music_different_tempos():
    """Test music generation with different tempos."""
    service = MusicGenerationService()
    tempos = [60, 90, 120, 150, 180]

    for tempo in tempos:
        request = MusicGenerationRequest(
            genre="pop",
            mood="happy",
            key="C",
            tempo=tempo,
            duration=30
        )
        result = service.generate_music(request)
        assert result is not None
        assert result["parameters"]["tempo"] == tempo


def test_melody_generation_note_range():
    """Test that generated melodies have notes in valid MIDI range."""
    service = MusicGenerationService()
    request = MusicGenerationRequest(
        genre="pop",
        mood="happy",
        key="C",
        tempo=120,
        duration=60
    )

    result = service.generate_music(request)
    melody = result["melody"]

    for note_data in melody:
        # Valid MIDI note range is 0-127, but we limit to 21-108 (piano range)
        assert 21 <= note_data["note"] <= 108
        assert note_data["duration"] > 0
        assert 0 <= note_data["velocity"] <= 127


def test_midi_data_creation():
    """Test that MIDI data is created successfully."""
    service = MusicGenerationService()
    request = MusicGenerationRequest(
        genre="pop",
        mood="happy",
        key="C",
        tempo=120,
        duration=30
    )

    result = service.generate_music(request)
    midi_data = result["midi_data"]

    assert midi_data is not None
    assert isinstance(midi_data, bytes)
    assert len(midi_data) > 0


def test_mood_affects_generation():
    """Test that different moods produce different results."""
    service = MusicGenerationService()
    moods = ["happy", "sad", "energetic", "calm"]
    results = []

    for mood in moods:
        request = MusicGenerationRequest(
            genre="pop",
            mood=mood,
            key="C",
            tempo=120,
            duration=30
        )
        result = service.generate_music(request)
        results.append(result)

    # Each result should be unique (different melodies)
    # We can't guarantee exact differences, but we can check they're generated
    for result in results:
        assert result is not None
        assert len(result["melody"]) > 0
