"""Backend package initialization"""
from backend.music_generator import MusicGenerator
from backend.lyric_generator import LyricGenerator
from backend.album_art_generator import AlbumArtGenerator

__all__ = ['MusicGenerator', 'LyricGenerator', 'AlbumArtGenerator']
