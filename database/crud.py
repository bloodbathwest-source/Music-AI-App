"""
CRUD operations for the Music AI App database
"""
from database.models import Song, Playlist, UserPreference, get_session
from datetime import datetime
from typing import List, Optional


class SongService:
    """Service for song CRUD operations"""
    
    @staticmethod
    def create_song(title: str, artist: str = "AI Generated", genre: str = None, 
                   duration: float = 0, file_path: str = None, lyrics: str = None,
                   album_art_path: str = None, key: str = None, tempo: int = None) -> Song:
        """Create a new song"""
        session = get_session()
        song = Song(
            title=title,
            artist=artist,
            genre=genre,
            duration=duration,
            file_path=file_path,
            lyrics=lyrics,
            album_art_path=album_art_path,
            key=key,
            tempo=tempo,
            is_ai_generated=True
        )
        session.add(song)
        session.commit()
        song_id = song.id
        session.close()
        return song_id
    
    @staticmethod
    def get_all_songs() -> List[Song]:
        """Get all songs"""
        session = get_session()
        songs = session.query(Song).order_by(Song.created_at.desc()).all()
        session.close()
        return songs
    
    @staticmethod
    def get_song_by_id(song_id: int) -> Optional[Song]:
        """Get song by ID"""
        session = get_session()
        song = session.query(Song).filter(Song.id == song_id).first()
        session.close()
        return song
    
    @staticmethod
    def search_songs(query: str) -> List[Song]:
        """Search songs by title, artist, or genre"""
        session = get_session()
        songs = session.query(Song).filter(
            (Song.title.contains(query)) |
            (Song.artist.contains(query)) |
            (Song.genre.contains(query))
        ).all()
        session.close()
        return songs
    
    @staticmethod
    def delete_song(song_id: int) -> bool:
        """Delete a song"""
        session = get_session()
        song = session.query(Song).filter(Song.id == song_id).first()
        if song:
            session.delete(song)
            session.commit()
            session.close()
            return True
        session.close()
        return False


class PlaylistService:
    """Service for playlist CRUD operations"""
    
    @staticmethod
    def create_playlist(name: str, description: str = None) -> int:
        """Create a new playlist"""
        session = get_session()
        playlist = Playlist(name=name, description=description, song_ids="")
        session.add(playlist)
        session.commit()
        playlist_id = playlist.id
        session.close()
        return playlist_id
    
    @staticmethod
    def get_all_playlists() -> List[Playlist]:
        """Get all playlists"""
        session = get_session()
        playlists = session.query(Playlist).order_by(Playlist.created_at.desc()).all()
        session.close()
        return playlists
    
    @staticmethod
    def add_song_to_playlist(playlist_id: int, song_id: int) -> bool:
        """Add a song to a playlist"""
        session = get_session()
        playlist = session.query(Playlist).filter(Playlist.id == playlist_id).first()
        if playlist:
            song_ids = playlist.song_ids.split(',') if playlist.song_ids else []
            if str(song_id) not in song_ids:
                song_ids.append(str(song_id))
                playlist.song_ids = ','.join(song_ids)
                session.commit()
            session.close()
            return True
        session.close()
        return False
    
    @staticmethod
    def get_playlist_songs(playlist_id: int) -> List[Song]:
        """Get all songs in a playlist"""
        session = get_session()
        playlist = session.query(Playlist).filter(Playlist.id == playlist_id).first()
        if playlist and playlist.song_ids:
            song_ids = [int(sid) for sid in playlist.song_ids.split(',') if sid]
            songs = session.query(Song).filter(Song.id.in_(song_ids)).all()
            session.close()
            return songs
        session.close()
        return []


class UserPreferenceService:
    """Service for user preference CRUD operations"""
    
    @staticmethod
    def get_theme() -> str:
        """Get theme preference"""
        session = get_session()
        prefs = session.query(UserPreference).first()
        if not prefs:
            theme = 'dark'
        else:
            theme = prefs.theme
        session.close()
        return theme
    
    @staticmethod
    def get_preferences() -> dict:
        """Get user preferences as dictionary"""
        session = get_session()
        prefs = session.query(UserPreference).first()
        if not prefs:
            prefs = UserPreference()
            session.add(prefs)
            session.commit()
        
        result = {
            'theme': prefs.theme,
            'default_genre': prefs.default_genre,
            'default_key': prefs.default_key,
            'default_tempo': prefs.default_tempo
        }
        session.close()
        return result
    
    @staticmethod
    def update_theme(theme: str) -> bool:
        """Update theme preference"""
        session = get_session()
        prefs = session.query(UserPreference).first()
        if not prefs:
            prefs = UserPreference(theme=theme)
            session.add(prefs)
        else:
            prefs.theme = theme
            prefs.last_updated = datetime.utcnow()
        session.commit()
        session.close()
        return True
    
    @staticmethod
    def update_defaults(genre: str = None, key: str = None, tempo: int = None) -> bool:
        """Update default music generation preferences"""
        session = get_session()
        prefs = session.query(UserPreference).first()
        if not prefs:
            prefs = UserPreference()
            session.add(prefs)
        
        if genre:
            prefs.default_genre = genre
        if key:
            prefs.default_key = key
        if tempo:
            prefs.default_tempo = tempo
        prefs.last_updated = datetime.utcnow()
        session.commit()
        session.close()
        return True
