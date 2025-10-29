"""Test cases for the Music AI App API."""
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint returns correct message."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Music AI App API"
    assert data["version"] == "1.0.0"


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_register_user():
    """Test user registration."""
    user_data = {
        "email": "test@example.com",
        "username": "testuser",
        "password": "testpass123"
    }
    response = client.post("/api/auth/register", json=user_data)
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_login():
    """Test user login."""
    login_data = {
        "username": "testuser",
        "password": "testpass123"
    }
    response = client.post("/api/auth/token", data=login_data)
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_music_generation_without_auth():
    """Test that music generation requires authentication."""
    generation_params = {
        "genre": "pop",
        "mood": "happy",
        "key": "C",
        "tempo": 120,
        "duration": 60
    }
    response = client.post("/api/music/generate", json=generation_params)
    assert response.status_code == 401


def test_music_generation_with_auth():
    """Test music generation with authentication."""
    # Login first
    login_data = {"username": "testuser", "password": "testpass123"}
    login_response = client.post("/api/auth/token", data=login_data)
    token = login_response.json()["access_token"]

    # Generate music
    generation_params = {
        "genre": "pop",
        "mood": "happy",
        "key": "C",
        "tempo": 120,
        "duration": 60
    }
    headers = {"Authorization": f"Bearer {token}"}
    response = client.post("/api/music/generate", json=generation_params, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert data["status"] == "queued"


def test_music_generation_invalid_duration():
    """Test music generation with invalid duration."""
    # Login first
    login_data = {"username": "testuser", "password": "testpass123"}
    login_response = client.post("/api/auth/token", data=login_data)
    token = login_response.json()["access_token"]

    # Try to generate with invalid duration
    generation_params = {
        "genre": "pop",
        "mood": "happy",
        "key": "C",
        "tempo": 120,
        "duration": 200  # Invalid: exceeds max of 180 seconds
    }
    headers = {"Authorization": f"Bearer {token}"}
    response = client.post("/api/music/generate", json=generation_params, headers=headers)
    assert response.status_code == 422  # Pydantic validation error


def test_export_track():
    """Test track export."""
    # Login first
    login_data = {"username": "testuser", "password": "testpass123"}
    login_response = client.post("/api/auth/token", data=login_data)
    token = login_response.json()["access_token"]

    # Export track
    export_params = {
        "track_id": "track_123",
        "format": "mp3",
        "quality": "high"
    }
    headers = {"Authorization": f"Bearer {token}"}
    response = client.post("/api/export/export", json=export_params, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["track_id"] == "track_123"
    assert data["format"] == "mp3"
    assert "download_url" in data


def test_export_unsupported_format():
    """Test export with unsupported format."""
    # Login first
    login_data = {"username": "testuser", "password": "testpass123"}
    login_response = client.post("/api/auth/token", data=login_data)
    token = login_response.json()["access_token"]

    # Try unsupported format
    export_params = {
        "track_id": "track_123",
        "format": "ogg",  # Unsupported
        "quality": "high"
    }
    headers = {"Authorization": f"Bearer {token}"}
    response = client.post("/api/export/export", json=export_params, headers=headers)
    assert response.status_code == 400


def test_get_user_profile():
    """Test getting user profile."""
    # Login first
    login_data = {"username": "testuser", "password": "testpass123"}
    login_response = client.post("/api/auth/token", data=login_data)
    token = login_response.json()["access_token"]

    # Get profile
    headers = {"Authorization": f"Bearer {token}"}
    response = client.get("/api/users/profile", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "email" in data
    assert "username" in data
    assert data["is_active"] is True
