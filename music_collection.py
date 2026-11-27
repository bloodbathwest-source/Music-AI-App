"""
Collection persistence and audio rendering for the Streamlit app (app.py).

Everything in app.py lives in ``st.session_state``, which evaporates when
the browser tab closes. This module provides the two pieces that fix that:

* ``melody_to_wav_bytes`` — render a melody (a list of ``(pitch, beats,
  velocity)`` tuples, the shape produced by app.py's genetic algorithm) to
  real WAV audio using pretty_midi's built-in synthesizer, so a track can
  be played in the browser and downloaded as WAV, not just MIDI.
* ``collection_to_json`` / ``parse_collection`` — serialize a user's track
  collection plus reward progress to a small JSON document, and validate /
  normalize such a document on the way back in. Parsing is strict: any
  malformed content raises :class:`CollectionFormatError`, so a corrupt
  upload can be rejected without touching the current session.

This module deliberately avoids importing streamlit or torch so the
round-trip logic can be unit-tested in the lightweight backend venv
(see tests/test_collection.py).
"""

import io
import json
import wave
from datetime import date, datetime

COLLECTION_FORMAT = "music-ai-app-collection"
COLLECTION_VERSION = 1

# user_stats fields (mirrors initialize_session_state in app.py)
_STAT_INT_FIELDS = (
    "tracks_created",
    "genres_explored",
    "keys_used",
    "modes_used",
    "emotions_used",
    "daily_streak",
    "max_generations",
    "total_actions",
)
_STAT_LIST_FIELDS = ("genres_list", "keys_list", "modes_list", "emotions_list")
# The matching counters are derived from the lists, as in app.py.
_LIST_TO_COUNT = {
    "genres_list": "genres_explored",
    "keys_list": "keys_used",
    "modes_list": "modes_used",
    "emotions_list": "emotions_used",
}


class CollectionFormatError(ValueError):
    """Raised when a collection file is missing, corrupt, or unsupported."""


# ============================================================================
# AUDIO RENDERING
# ============================================================================

def melody_to_wav_bytes(melody, tempo: int = 120, sample_rate: int = 22050) -> bytes:
    """Render a melody to WAV bytes using pretty_midi's synthesizer.

    ``melody`` is a sequence of ``(pitch, duration_in_beats, velocity)``
    notes played one after another, exactly as app.py's
    ``create_midi_from_melody`` interprets them. The returned bytes are a
    complete 16-bit mono WAV file containing the actual notes.
    """
    import numpy as np
    import pretty_midi

    notes = _validate_melody(melody, "melody")
    if isinstance(tempo, bool) or not isinstance(tempo, (int, float)) or tempo <= 0:
        raise CollectionFormatError("tempo must be a positive number")

    pm = pretty_midi.PrettyMIDI(initial_tempo=float(tempo))
    instrument = pretty_midi.Instrument(program=0)  # acoustic grand piano
    seconds_per_beat = 60.0 / float(tempo)
    start = 0.0
    for pitch, duration, velocity in notes:
        duration_seconds = float(duration) * seconds_per_beat
        instrument.notes.append(
            pretty_midi.Note(
                velocity=int(velocity),
                pitch=int(pitch),
                start=start,
                end=start + max(duration_seconds, 0.05),
            )
        )
        start += duration_seconds
    pm.instruments.append(instrument)

    audio = pm.synthesize(fs=sample_rate)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return buffer.getvalue()


# ============================================================================
# SERIALIZATION
# ============================================================================

def collection_to_json(*, tracks, user_stats, achievements, titles,
                       collectibles, selected_title, rewards_enabled) -> bytes:
    """Serialize a collection + reward progress to JSON bytes.

    Inputs mirror app.py's session state:

    * ``tracks``: list of track dicts (see ``parse_collection`` for shape)
    * ``user_stats``: the stats dict (sets and dates are converted)
    * ``achievements``: list of ``{"id", "unlocked", "unlock_date"}`` dicts
    * ``titles``: list of ``{"id", "unlocked"}`` dicts
    * ``collectibles``: list of collectible id strings (duplicates kept —
      a user can own the same collectible twice)
    """
    document = {
        "format": COLLECTION_FORMAT,
        "version": COLLECTION_VERSION,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "tracks": [_serialize_track(track) for track in tracks],
        "user_stats": _serialize_user_stats(user_stats),
        "achievements": [
            {
                "id": entry["id"],
                "unlocked": bool(entry["unlocked"]),
                "unlock_date": _iso_or_none(entry.get("unlock_date")),
            }
            for entry in achievements
        ],
        "titles": [
            {"id": entry["id"], "unlocked": bool(entry["unlocked"])}
            for entry in titles
        ],
        "collectibles": [str(collectible_id) for collectible_id in collectibles],
        "selected_title": selected_title,
        "rewards_enabled": bool(rewards_enabled),
    }
    return json.dumps(document, indent=2).encode("utf-8")


def parse_collection(raw) -> dict:
    """Validate and normalize a collection document.

    Accepts bytes or str. Returns a plain-data dict with keys ``tracks``,
    ``user_stats``, ``achievements`` (id -> {"unlocked", "unlock_date"}),
    ``titles`` (id -> bool), ``collectibles``, ``selected_title`` and
    ``rewards_enabled``. Raises :class:`CollectionFormatError` on any
    problem; the caller's state should be left untouched in that case.
    """
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = bytes(raw).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CollectionFormatError(f"file is not valid UTF-8 text: {exc}") from exc
    if not isinstance(raw, str):
        raise CollectionFormatError("collection data must be text or bytes")

    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CollectionFormatError(f"file is not valid JSON: {exc}") from exc

    if not isinstance(document, dict):
        raise CollectionFormatError("collection file must contain a JSON object")
    if document.get("format") != COLLECTION_FORMAT:
        raise CollectionFormatError("this file is not a Music AI App collection")

    version = document.get("version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise CollectionFormatError("collection file has no valid version number")
    if version > COLLECTION_VERSION:
        raise CollectionFormatError(
            f"collection was saved by a newer version of the app (format v{version})"
        )
    if version < 1:
        raise CollectionFormatError(f"unsupported collection format version: {version}")

    tracks_raw = document.get("tracks", [])
    if not isinstance(tracks_raw, list):
        raise CollectionFormatError("'tracks' must be a list")
    tracks = [_validate_track(track, i) for i, track in enumerate(tracks_raw)]

    collectibles_raw = document.get("collectibles", [])
    if not isinstance(collectibles_raw, list) or \
            not all(isinstance(c, str) for c in collectibles_raw):
        raise CollectionFormatError("'collectibles' must be a list of collectible ids")

    selected_title = document.get("selected_title")
    if selected_title is not None and not isinstance(selected_title, str):
        raise CollectionFormatError("'selected_title' must be a string or null")

    rewards_enabled = document.get("rewards_enabled", True)
    if not isinstance(rewards_enabled, bool):
        raise CollectionFormatError("'rewards_enabled' must be true or false")

    return {
        "version": version,
        "tracks": tracks,
        "user_stats": _validate_user_stats(document.get("user_stats")),
        "achievements": _validate_achievements(document.get("achievements")),
        "titles": _validate_titles(document.get("titles")),
        "collectibles": list(collectibles_raw),
        "selected_title": selected_title,
        "rewards_enabled": rewards_enabled,
    }


# ============================================================================
# VALIDATION / CONVERSION HELPERS
# ============================================================================

def _iso_or_none(value):
    """Convert a date/datetime/str/None to an ISO string or None."""
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def _validate_melody(melody, where):
    """Validate a melody, returning notes as [pitch, duration, velocity]."""
    if not isinstance(melody, (list, tuple)) or len(melody) == 0:
        raise CollectionFormatError(f"{where}: expected a non-empty list of notes")
    notes = []
    for i, note in enumerate(melody):
        if not isinstance(note, (list, tuple)) or len(note) != 3:
            raise CollectionFormatError(
                f"{where}[{i}]: each note must be [pitch, duration, velocity]"
            )
        pitch, duration, velocity = note
        for label, value in (("pitch", pitch), ("duration", duration),
                             ("velocity", velocity)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise CollectionFormatError(f"{where}[{i}]: {label} must be a number")
        if not 0 <= pitch <= 127:
            raise CollectionFormatError(f"{where}[{i}]: pitch {pitch} out of range 0-127")
        if not 0 < duration <= 64:
            raise CollectionFormatError(
                f"{where}[{i}]: duration {duration} out of range (0, 64] beats"
            )
        if not 0 <= velocity <= 127:
            raise CollectionFormatError(
                f"{where}[{i}]: velocity {velocity} out of range 0-127"
            )
        notes.append([int(pitch), duration, int(velocity)])
    return notes


def _serialize_track(track):
    return {
        "name": track.get("name", ""),
        "melody": [list(note) for note in track["melody"]],
        "genre": track.get("genre", ""),
        "key_root": track.get("key_root", "C"),
        "mode": track.get("mode", "major"),
        "emotion": track.get("emotion", ""),
        "generations": track.get("generations", 20),
        "tempo": track.get("tempo", 120),
        "created_at": _iso_or_none(track.get("created_at")) or "",
    }


def _validate_track(track, index):
    where = f"tracks[{index}]"
    if not isinstance(track, dict):
        raise CollectionFormatError(f"{where}: each track must be an object")

    def _text(key, default=""):
        value = track.get(key, default)
        return value if isinstance(value, str) else default

    def _bounded_int(key, default, low, high):
        value = track.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return default
        return max(low, min(high, int(value)))

    return {
        "name": _text("name") or f"Track {index + 1}",
        "melody": _validate_melody(track.get("melody"), f"{where}.melody"),
        "genre": _text("genre"),
        "key_root": _text("key_root", "C"),
        "mode": _text("mode", "major"),
        "emotion": _text("emotion"),
        "generations": _bounded_int("generations", 20, 0, 10000),
        "tempo": _bounded_int("tempo", 120, 30, 300),
        "created_at": _text("created_at"),
    }


def _serialize_user_stats(stats):
    serialized = {}
    for field in _STAT_INT_FIELDS:
        serialized[field] = int(stats.get(field, 0))
    for field in _STAT_LIST_FIELDS:
        serialized[field] = sorted(str(item) for item in stats.get(field, ()))
    serialized["last_login"] = _iso_or_none(stats.get("last_login"))
    return serialized


def _validate_user_stats(stats):
    if stats is None:
        stats = {}
    if not isinstance(stats, dict):
        raise CollectionFormatError("'user_stats' must be an object")

    validated = {}
    for field in _STAT_INT_FIELDS:
        value = stats.get(field, 0)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CollectionFormatError(f"user_stats.{field} must be a number")
        validated[field] = int(value)
    for field in _STAT_LIST_FIELDS:
        value = stats.get(field, [])
        if not isinstance(value, list) or \
                not all(isinstance(item, str) for item in value):
            raise CollectionFormatError(f"user_stats.{field} must be a list of strings")
        validated[field] = list(dict.fromkeys(value))  # dedupe, keep order
        # Keep the derived counters consistent with the lists, as app.py does.
        validated[_LIST_TO_COUNT[field]] = len(validated[field])

    last_login = stats.get("last_login")
    if last_login is not None:
        if not isinstance(last_login, str):
            raise CollectionFormatError("user_stats.last_login must be a date string or null")
        try:
            date.fromisoformat(last_login)
        except ValueError:
            try:
                datetime.fromisoformat(last_login)
                last_login = last_login[:10]
                date.fromisoformat(last_login)
            except ValueError as exc:
                raise CollectionFormatError(
                    f"user_stats.last_login is not a valid date: {last_login!r}"
                ) from exc
    validated["last_login"] = last_login
    return validated


def _validate_achievements(entries):
    if entries is None:
        return {}
    if not isinstance(entries, list):
        raise CollectionFormatError("'achievements' must be a list")
    validated = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise CollectionFormatError("each achievement must be an object with an 'id'")
        unlocked = entry.get("unlocked", False)
        if not isinstance(unlocked, bool):
            raise CollectionFormatError(
                f"achievements[{entry['id']}].unlocked must be true or false"
            )
        unlock_date = entry.get("unlock_date")
        if unlock_date is not None:
            if not isinstance(unlock_date, str):
                raise CollectionFormatError(
                    f"achievements[{entry['id']}].unlock_date must be a string or null"
                )
            try:
                datetime.fromisoformat(unlock_date)
            except ValueError as exc:
                raise CollectionFormatError(
                    f"achievements[{entry['id']}].unlock_date is not a valid timestamp"
                ) from exc
        validated[entry["id"]] = {"unlocked": unlocked, "unlock_date": unlock_date}
    return validated


def _validate_titles(entries):
    if entries is None:
        return {}
    if not isinstance(entries, list):
        raise CollectionFormatError("'titles' must be a list")
    validated = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise CollectionFormatError("each title must be an object with an 'id'")
        unlocked = entry.get("unlocked", False)
        if not isinstance(unlocked, bool):
            raise CollectionFormatError(f"titles[{entry['id']}].unlocked must be true or false")
        validated[entry["id"]] = unlocked
    return validated
