"""In-memory store for music generation tasks and generated tracks.

The music and export endpoints originally returned mock data: generation
"queued" a task that never ran, status always reported a completed fake
track, and the track list was always empty. This module holds the real
records produced when POST /api/music/generate runs the generation service
as a background task, so status polling, track listing, sharing, and MIDI
download all operate on actual generated music.

Like user_store, records live for the lifetime of the process; durable
persistence belongs in the database layer (backend/app/core/database.py),
and these functions are the interface such an implementation would keep.
"""
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

_tasks: Dict[str, dict] = {}
_tracks: Dict[str, dict] = {}


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _public_track(track: dict) -> dict:
    """Return the track record without internal fields (owner, MIDI bytes)."""
    return {
        "id": track["id"],
        "title": track["title"],
        "genre": track["genre"],
        "duration": track["duration"],
        "file_path": track["file_path"],
        "created_at": track["created_at"],
        "is_public": track["is_public"],
    }


# ---------------------------------------------------------------------------
# Generation tasks
# ---------------------------------------------------------------------------

def create_task(owner: str, request) -> dict:
    """Create a queued generation task for a MusicGenerationRequest."""
    task = {
        "id": _new_id("task"),
        "owner": owner,
        "request": request,
        "status": "queued",
        "progress": 0,
        "result": None,
        "error": None,
    }
    _tasks[task["id"]] = task
    return task


def get_task(task_id: str) -> Optional[dict]:
    """Return the task record (including owner and request), or None."""
    return _tasks.get(task_id)


def complete_task(task_id: str, track: dict) -> None:
    """Mark a task completed and point it at the generated track."""
    task = _tasks.get(task_id)
    if task is None:
        return
    task["status"] = "completed"
    task["progress"] = 100
    task["result"] = {
        "track_id": track["id"],
        "file_path": track["file_path"],
    }


def fail_task(task_id: str, error: str) -> None:
    """Mark a task failed with an error message."""
    task = _tasks.get(task_id)
    if task is None:
        return
    task["status"] = "failed"
    task["error"] = error


# ---------------------------------------------------------------------------
# Tracks
# ---------------------------------------------------------------------------

def add_track(owner: str, genre: str, duration: int, midi_data: bytes,
              parameters: dict) -> dict:
    """Store a generated track and return its public record."""
    track_id = _new_id("track")
    track = {
        "id": track_id,
        "owner": owner,
        "title": f"Generated {genre} track",
        "genre": genre,
        "duration": duration,
        "file_path": f"/tracks/{track_id}.mid",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "is_public": False,
        "midi_data": midi_data,
        "parameters": parameters,
    }
    _tracks[track_id] = track
    return _public_track(track)


def get_track(track_id: str) -> Optional[dict]:
    """Return the full track record (including owner and MIDI data), or None."""
    return _tracks.get(track_id)


def get_public_track(track_id: str) -> Optional[dict]:
    """Return the public track record, or None."""
    track = _tracks.get(track_id)
    return _public_track(track) if track else None


def list_tracks(owner: str, skip: int = 0, limit: int = 10) -> List[dict]:
    """Return a page of the owner's tracks, oldest first."""
    owned = [t for t in _tracks.values() if t["owner"] == owner]
    return [_public_track(t) for t in owned[skip:skip + limit]]


def list_all_tracks(owner: str) -> List[dict]:
    """Return all of the owner's tracks (public records)."""
    return [_public_track(t) for t in _tracks.values() if t["owner"] == owner]


def delete_track(owner: str, track_id: str) -> bool:
    """Delete a track owned by the user. Returns True if it existed."""
    track = _tracks.get(track_id)
    if track is None or track["owner"] != owner:
        return False
    del _tracks[track_id]
    return True


def set_track_public(owner: str, track_id: str, is_public: bool) -> Optional[dict]:
    """Set a track's sharing flag. Returns the public record, or None."""
    track = _tracks.get(track_id)
    if track is None or track["owner"] != owner:
        return None
    track["is_public"] = is_public
    return _public_track(track)
