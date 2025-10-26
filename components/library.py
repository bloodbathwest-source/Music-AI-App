"""
Library component for displaying and managing music collection
"""
import streamlit as st
from database.crud import SongService, PlaylistService
from typing import List
import os


def render_library():
    """Render the music library component"""
    st.markdown("### 📚 Music Library")
    
    # Tabs for songs and playlists
    tab1, tab2 = st.tabs(["🎵 Songs", "📋 Playlists"])
    
    with tab1:
        render_songs_library()
    
    with tab2:
        render_playlists_library()


def render_songs_library():
    """Render songs library"""
    songs = SongService.get_all_songs()
    
    if not songs:
        st.info("Your library is empty. Generate some music to get started!")
        return
    
    # Display controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"**Total Songs: {len(songs)}**")
    
    with col2:
        sort_by = st.selectbox(
            "Sort by",
            ["Recent", "Title", "Genre"],
            label_visibility="collapsed"
        )
    
    with col3:
        view_mode = st.radio(
            "View",
            ["Grid", "List"],
            horizontal=True,
            label_visibility="collapsed"
        )
    
    # Sort songs
    if sort_by == "Title":
        songs = sorted(songs, key=lambda x: x.title)
    elif sort_by == "Genre":
        songs = sorted(songs, key=lambda x: x.genre or "")
    
    st.markdown("---")
    
    # Display songs
    if view_mode == "Grid":
        render_songs_grid(songs)
    else:
        render_songs_list(songs)


def render_songs_grid(songs: List):
    """Render songs in grid view"""
    cols_per_row = 3
    
    for i in range(0, len(songs), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, col in enumerate(cols):
            if i + j < len(songs):
                song = songs[i + j]
                
                with col:
                    with st.container():
                        # Album art
                        if song.album_art_path and os.path.exists(song.album_art_path):
                            st.image(song.album_art_path, use_container_width=True)
                        else:
                            st.image("https://via.placeholder.com/200?text=No+Art", use_container_width=True)
                        
                        # Song info
                        st.markdown(f"**{song.title[:30]}**")
                        st.caption(f"{song.artist}")
                        
                        if song.genre:
                            st.caption(f"🎸 {song.genre}")
                        
                        # Actions
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            if st.button("▶️ Play", key=f"play_{song.id}"):
                                st.session_state.current_song_id = song.id
                                st.rerun()
                        
                        with col_b:
                            if st.button("🗑️", key=f"delete_{song.id}"):
                                SongService.delete_song(song.id)
                                st.rerun()
                        
                        st.markdown("---")


def render_songs_list(songs: List):
    """Render songs in list view"""
    for song in songs:
        col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 1, 1])
        
        with col1:
            st.markdown(f"**{song.title}**")
        
        with col2:
            st.text(song.artist)
        
        with col3:
            st.text(song.genre or "Unknown")
        
        with col4:
            if st.button("▶️", key=f"play_list_{song.id}"):
                st.session_state.current_song_id = song.id
                st.rerun()
        
        with col5:
            if st.button("🗑️", key=f"delete_list_{song.id}"):
                SongService.delete_song(song.id)
                st.rerun()


def render_playlists_library():
    """Render playlists library"""
    playlists = PlaylistService.get_all_playlists()
    
    # Create new playlist
    with st.expander("➕ Create New Playlist"):
        playlist_name = st.text_input("Playlist Name")
        playlist_desc = st.text_area("Description (optional)")
        
        if st.button("Create Playlist"):
            if playlist_name:
                PlaylistService.create_playlist(playlist_name, playlist_desc)
                st.success(f"Created playlist: {playlist_name}")
                st.rerun()
            else:
                st.error("Please enter a playlist name")
    
    st.markdown("---")
    
    # Display playlists
    if not playlists:
        st.info("No playlists yet. Create one to organize your music!")
        return
    
    for playlist in playlists:
        with st.expander(f"📋 {playlist.name}"):
            st.markdown(f"*{playlist.description or 'No description'}*")
            
            songs = PlaylistService.get_playlist_songs(playlist.id)
            
            if songs:
                st.markdown(f"**{len(songs)} songs**")
                for song in songs:
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.text(f"♪ {song.title} - {song.artist}")
                    with col2:
                        if st.button("▶️", key=f"play_pl_{playlist.id}_{song.id}"):
                            st.session_state.current_song_id = song.id
                            st.rerun()
            else:
                st.info("This playlist is empty")
