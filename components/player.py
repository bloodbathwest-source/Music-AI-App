"""
Player component for music playback
"""
import streamlit as st
from typing import Optional
import os


def render_player(current_song: Optional[dict] = None):
    """Render the music player component"""
    st.markdown("### 🎵 Now Playing")
    
    if current_song:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Album art
            if current_song.get('album_art_path') and os.path.exists(current_song['album_art_path']):
                st.image(current_song['album_art_path'], width=200)
            else:
                st.image("https://via.placeholder.com/200?text=No+Art", width=200)
        
        with col2:
            # Song info
            st.markdown(f"**{current_song.get('title', 'Unknown Title')}**")
            st.markdown(f"*{current_song.get('artist', 'Unknown Artist')}*")
            
            if current_song.get('genre'):
                st.caption(f"🎸 {current_song['genre']}")
            
            if current_song.get('key') and current_song.get('tempo'):
                st.caption(f"🎼 Key: {current_song['key']} | Tempo: {current_song['tempo']} BPM")
        
        # Audio player
        if current_song.get('file_path') and os.path.exists(current_song['file_path']):
            with open(current_song['file_path'], 'rb') as audio_file:
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format='audio/midi')
        
        # Playback controls (visual only with Streamlit)
        st.markdown("---")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if st.button("⏮️ Previous"):
                st.session_state.player_action = 'previous'
        
        with col2:
            if st.button("⏯️ Play/Pause"):
                st.session_state.player_action = 'play_pause'
        
        with col3:
            if st.button("⏭️ Next"):
                st.session_state.player_action = 'next'
        
        with col4:
            if st.button("🔀 Shuffle"):
                st.session_state.shuffle = not st.session_state.get('shuffle', False)
        
        with col5:
            if st.button("🔁 Repeat"):
                st.session_state.repeat = not st.session_state.get('repeat', False)
        
        # Display lyrics if available
        if current_song.get('lyrics'):
            with st.expander("📝 View Lyrics"):
                st.text(current_song['lyrics'])
    
    else:
        st.info("No song currently playing. Select a song from your library or generate a new one!")
        
        # Placeholder controls
        col1, col2, col3 = st.columns(3)
        with col1:
            st.button("⏮️ Previous", disabled=True)
        with col2:
            st.button("⏯️ Play/Pause", disabled=True)
        with col3:
            st.button("⏭️ Next", disabled=True)
