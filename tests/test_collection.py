"""Tests for music_collection: WAV rendering and collection save/load.

These cover the Streamlit app's (app.py) persistence helpers, which live in
music_collection.py precisely so they can be tested without importing
streamlit or torch.
"""

import io
import json
import wave
from datetime import date, datetime

import pytest

from music_collection import (
    COLLECTION_FORMAT,
    CollectionFormatError,
    collection_to_json,
    melody_to_wav_bytes,
    parse_collection,
)

# 5 beats total; at 120 BPM that is 2.5 seconds of audio.
MELODY = [(72, 1, 100), (74, 1, 90), (76, 2, 110), (79, 1, 100)]


def _sample_stats():
    return {
        "tracks_created": 3,
        "genres_explored": 2,
        "keys_used": 2,
        "modes_used": 1,
        "emotions_used": 1,
        "daily_streak": 2,
        "max_generations": 50,
        "total_actions": 3,
        "last_login": date(2026, 11, 26),
        "genres_list": {"pop", "jazz"},
        "keys_list": {"C", "G"},
        "modes_list": {"major"},
        "emotions_list": {"happy"},
    }


def _sample_export() -> bytes:
    return collection_to_json(
        tracks=[{
            "name": "Pop — C major (happy)",
            "melody": MELODY,
            "genre": "pop",
            "key_root": "C",
            "mode": "major",
            "emotion": "happy",
            "generations": 20,
            "tempo": 120,
            "created_at": "2026-11-26T20:00:00",
        }],
        user_stats=_sample_stats(),
        achievements=[
            {"id": "first_track", "unlocked": True,
             "unlock_date": datetime(2026, 11, 26, 20, 0, 0)},
            {"id": "track_master", "unlocked": False, "unlock_date": None},
        ],
        titles=[{"id": "lofi_legend", "unlocked": True}],
        # Duplicates are meaningful: the same collectible can be owned twice.
        collectibles=["note_sticker", "fire_badge", "note_sticker"],
        selected_title="Lofi Legend 🎧",
        rewards_enabled=True,
    )


class TestCollectionRoundTrip:
    def test_round_trip_preserves_everything(self):
        doc = parse_collection(_sample_export())

        track = doc["tracks"][0]
        assert track["melody"] == [[72, 1, 100], [74, 1, 90],
                                   [76, 2, 110], [79, 1, 100]]
        assert track["genre"] == "pop"
        assert track["tempo"] == 120
        assert track["created_at"] == "2026-11-26T20:00:00"

        stats = doc["user_stats"]
        assert stats["tracks_created"] == 3
        assert sorted(stats["genres_list"]) == ["jazz", "pop"]
        assert stats["genres_explored"] == 2
        assert stats["keys_used"] == 2
        assert stats["last_login"] == "2026-11-26"

        assert doc["achievements"]["first_track"] == {
            "unlocked": True, "unlock_date": "2026-11-26T20:00:00"}
        assert doc["achievements"]["track_master"] == {
            "unlocked": False, "unlock_date": None}
        assert doc["titles"] == {"lofi_legend": True}
        assert doc["collectibles"] == ["note_sticker", "fire_badge",
                                        "note_sticker"]
        assert doc["selected_title"] == "Lofi Legend 🎧"
        assert doc["rewards_enabled"] is True

    def test_export_is_tagged_json(self):
        raw = json.loads(_sample_export())
        assert raw["format"] == COLLECTION_FORMAT
        assert raw["version"] == 1

    def test_empty_collection_round_trip(self):
        stats = _sample_stats()
        stats.update({field: 0 for field in
                      ("tracks_created", "genres_explored", "keys_used",
                       "modes_used", "emotions_used", "daily_streak",
                       "max_generations", "total_actions")})
        stats.update({field: set() for field in
                      ("genres_list", "keys_list", "modes_list",
                       "emotions_list")})
        stats["last_login"] = None
        doc = parse_collection(collection_to_json(
            tracks=[], user_stats=stats, achievements=[], titles=[],
            collectibles=[], selected_title=None, rewards_enabled=False))
        assert doc["tracks"] == []
        assert doc["user_stats"]["last_login"] is None
        assert doc["selected_title"] is None
        assert doc["rewards_enabled"] is False


class TestParseRejectsBadFiles:
    def test_rejects_non_json(self):
        with pytest.raises(CollectionFormatError):
            parse_collection(b"this is not json at all")

    def test_rejects_non_utf8(self):
        with pytest.raises(CollectionFormatError):
            parse_collection(b"\xff\xfe\x00bad")

    def test_rejects_wrong_format_marker(self):
        with pytest.raises(CollectionFormatError):
            parse_collection(b'{"format": "someone-elses-file", "version": 1}')

    def test_rejects_newer_version(self):
        payload = json.dumps({"format": COLLECTION_FORMAT, "version": 99})
        with pytest.raises(CollectionFormatError):
            parse_collection(payload.encode())

    def test_rejects_top_level_list(self):
        with pytest.raises(CollectionFormatError):
            parse_collection(b"[1, 2, 3]")

    def test_rejects_out_of_range_pitch(self):
        doc = json.loads(_sample_export())
        doc["tracks"][0]["melody"] = [[999, 1, 100]]
        with pytest.raises(CollectionFormatError):
            parse_collection(json.dumps(doc).encode())

    def test_rejects_non_list_melody(self):
        doc = json.loads(_sample_export())
        doc["tracks"][0]["melody"] = "not-a-melody"
        with pytest.raises(CollectionFormatError):
            parse_collection(json.dumps(doc).encode())

    def test_rejects_bad_stats(self):
        doc = json.loads(_sample_export())
        doc["user_stats"]["tracks_created"] = "many"
        with pytest.raises(CollectionFormatError):
            parse_collection(json.dumps(doc).encode())


class TestWavRendering:
    def test_wav_is_real_playable_audio(self):
        wav_bytes = melody_to_wav_bytes(MELODY, tempo=120)
        assert wav_bytes[:4] == b"RIFF"
        assert wav_bytes[8:12] == b"WAVE"
        with wave.open(io.BytesIO(wav_bytes)) as wav_file:
            assert wav_file.getnchannels() == 1
            assert wav_file.getsampwidth() == 2
            assert wav_file.getframerate() == 22050
            frames = wav_file.readframes(wav_file.getnframes())
            # 5 beats at 120 BPM = 2.5 s = 55125 frames at 22050 Hz.
            assert wav_file.getnframes() >= 22050 * 2
            # The actual notes must be in there — not silence.
            assert any(byte != 0 for byte in frames)

    def test_wav_duration_scales_with_tempo(self):
        slow = melody_to_wav_bytes(MELODY, tempo=60)
        fast = melody_to_wav_bytes(MELODY, tempo=120)
        assert len(slow) > len(fast)

    def test_wav_rejects_empty_melody(self):
        with pytest.raises(CollectionFormatError):
            melody_to_wav_bytes([], tempo=120)
