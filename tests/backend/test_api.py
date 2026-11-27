"""Test cases for the Music AI App API."""
import uuid

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def _unique_credentials():
    """Fresh username/email/password so tests never collide in the store."""
    suffix = uuid.uuid4().hex[:8]
    return {
        "email": f"user_{suffix}@example.com",
        "username": f"user_{suffix}",
        "password": "testpass123",
    }


def _register_and_login(**overrides):
    """Register a fresh user and return (credentials, auth headers)."""
    credentials = _unique_credentials()
    credentials.update(overrides)
    response = client.post("/api/auth/register", json=credentials)
    assert response.status_code == 200
    token = response.json()["access_token"]
    return credentials, {"Authorization": f"Bearer {token}"}


def _generate_track_id(headers):
    """Generate a track through the API and return its track_id."""
    params = {
        "genre": "pop",
        "mood": "happy",
        "key": "C",
        "tempo": 120,
        "duration": 30
    }
    response = client.post("/api/music/generate", json=params, headers=headers)
    assert response.status_code == 200
    task_id = response.json()["task_id"]
    # Background generation runs inside the request cycle under TestClient,
    # but poll briefly so the test does not depend on that timing.
    for _ in range(10):
        status = client.get(f"/api/music/generate/{task_id}", headers=headers)
        assert status.status_code == 200
        if status.json()["status"] == "completed":
            return status.json()["result"]["track_id"]
    raise AssertionError("Generation task did not complete")


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
    # MusicGenerationRequest.duration is Field(ge=30, le=180), so FastAPI
    # rejects out-of-range durations with 422 before the handler runs.
    assert response.status_code == 422


def test_export_track():
    """Test track export."""
    _, headers = _register_and_login()
    track_id = _generate_track_id(headers)

    # Export the real generated track
    export_params = {
        "track_id": track_id,
        "format": "mp3",
        "quality": "high"
    }
    response = client.post("/api/export/export", json=export_params, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["track_id"] == track_id
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


def test_login_wrong_password():
    """Test that login rejects a wrong password (auth used to accept any)."""
    credentials, _ = _register_and_login()
    response = client.post(
        "/api/auth/token",
        data={"username": credentials["username"], "password": "wrong-password"}
    )
    assert response.status_code == 401


def test_login_unknown_user():
    """Test that login rejects a user who never registered."""
    response = client.post(
        "/api/auth/token",
        data={"username": "never_registered", "password": "testpass123"}
    )
    assert response.status_code == 401


def test_register_duplicate_username():
    """Test that registering the same username twice is rejected."""
    credentials, _ = _register_and_login()
    response = client.post("/api/auth/register", json=credentials)
    assert response.status_code == 400


def test_profile_reflects_registered_user():
    """Profile and /me return the actual registered user, not a placeholder."""
    credentials, headers = _register_and_login(full_name="Casey Composer")

    me = client.get("/api/auth/me", headers=headers)
    assert me.status_code == 200
    assert me.json()["username"] == credentials["username"]
    assert me.json()["email"] == credentials["email"]
    assert me.json()["full_name"] == "Casey Composer"

    profile = client.get("/api/users/profile", headers=headers)
    assert profile.status_code == 200
    assert profile.json()["username"] == credentials["username"]
    assert profile.json()["full_name"] == "Casey Composer"

    updated = client.put(
        "/api/users/profile",
        json={"full_name": "Casey C. Composer"},
        headers=headers
    )
    assert updated.status_code == 200
    assert updated.json()["full_name"] == "Casey C. Composer"


def test_generation_flow_end_to_end():
    """Generate -> status -> list -> get -> share -> download -> delete."""
    _, headers = _register_and_login()
    track_id = _generate_track_id(headers)

    # The track appears in the user's library
    tracks = client.get("/api/music/tracks", headers=headers)
    assert tracks.status_code == 200
    assert [t["id"] for t in tracks.json()] == [track_id]

    user_tracks = client.get("/api/users/tracks", headers=headers)
    assert user_tracks.json()["total"] == 1

    # Track details
    track = client.get(f"/api/music/tracks/{track_id}", headers=headers)
    assert track.status_code == 200
    assert track.json()["genre"] == "pop"

    # Sharing via the JSON body the frontend sends
    shared = client.post(
        f"/api/music/tracks/{track_id}/share",
        json={"is_public": True},
        headers=headers
    )
    assert shared.status_code == 200
    assert shared.json()["is_public"] is True

    # MIDI download returns real MIDI bytes (header magic "MThd")
    download = client.get(f"/api/export/download/{track_id}.mid", headers=headers)
    assert download.status_code == 200
    assert download.content[:4] == b"MThd"

    # Delete removes it
    deleted = client.delete(f"/api/music/tracks/{track_id}", headers=headers)
    assert deleted.status_code == 200
    gone = client.get(f"/api/music/tracks/{track_id}", headers=headers)
    assert gone.status_code == 404


def test_generation_status_unknown_task():
    """Status for a task that does not exist is a 404, not a fake success."""
    _, headers = _register_and_login()
    response = client.get("/api/music/generate/task_nope", headers=headers)
    assert response.status_code == 404


def test_get_unknown_track():
    """Getting a track that does not exist is a 404."""
    _, headers = _register_and_login()
    response = client.get("/api/music/tracks/track_nope", headers=headers)
    assert response.status_code == 404


def test_tracks_are_private_to_their_owner():
    """Another user cannot see or fetch someone else's unshared track."""
    _, owner_headers = _register_and_login()
    track_id = _generate_track_id(owner_headers)

    _, other_headers = _register_and_login()
    assert client.get("/api/music/tracks", headers=other_headers).json() == []
    response = client.get(f"/api/music/tracks/{track_id}", headers=other_headers)
    assert response.status_code == 404


def test_export_unknown_track():
    """Exporting a track that does not exist is a 404."""
    _, headers = _register_and_login()
    response = client.post(
        "/api/export/export",
        json={"track_id": "track_nope", "format": "midi", "quality": "high"},
        headers=headers
    )
    assert response.status_code == 404


def test_download_audio_format_not_renderable():
    """Audio downloads honestly report 501 (only MIDI can be rendered)."""
    _, headers = _register_and_login()
    track_id = _generate_track_id(headers)
    response = client.get(f"/api/export/download/{track_id}.mp3", headers=headers)
    assert response.status_code == 501
