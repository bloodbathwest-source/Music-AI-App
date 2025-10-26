"""
AI Creator component for generating music, lyrics, and album art
"""
import streamlit as st
from backend.music_generator import MusicGenerator
from backend.lyric_generator import LyricGenerator
from backend.album_art_generator import AlbumArtGenerator
from database.crud import SongService
import os
import re
from datetime import datetime


def render_ai_creator():
    """Render the AI music creator component"""
    st.markdown("### 🤖 AI Music Creator")
    st.markdown("Generate original music, lyrics, and album art using AI")
    
    # Creation mode tabs
    tab1, tab2, tab3 = st.tabs(["🎵 Full Song", "🎼 Music Only", "📝 Lyrics Only"])
    
    with tab1:
        render_full_song_creator()
    
    with tab2:
        render_music_only_creator()
    
    with tab3:
        render_lyrics_only_creator()


def render_full_song_creator():
    """Render full song creation interface"""
    st.markdown("#### Create a Complete AI-Generated Song")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Song details
        title = st.text_input("Song Title", value=f"AI Song {datetime.now().strftime('%Y%m%d_%H%M')}")
        genre = st.selectbox("Genre", ["pop", "rock", "jazz", "classical", "electronic"])
        emotion = st.selectbox("Emotion", ["happy", "sad", "energetic", "calm", "suspenseful"])
    
    with col2:
        # Musical parameters
        key = st.selectbox("Key", ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])
        mode = st.selectbox("Mode", ["major", "minor", "dorian", "phrygian", "mixolydian"])
        tempo = st.slider("Tempo (BPM)", 60, 180, 120)
    
    # Advanced options
    with st.expander("🎨 Advanced Options"):
        num_bars = st.slider("Number of Bars", 4, 32, 8)
        generate_lyrics = st.checkbox("Generate Lyrics", value=True)
        generate_album_art = st.checkbox("Generate Album Art", value=True)
    
    # Generate button
    if st.button("🎵 Generate Full Song", use_container_width=True, type="primary"):
        with st.spinner("Creating your song... This may take a moment..."):
            generate_full_song(
                title, genre, emotion, key, mode, tempo,
                num_bars, generate_lyrics, generate_album_art
            )


def render_music_only_creator():
    """Render music-only creation interface"""
    st.markdown("#### Generate Music Only")
    
    col1, col2 = st.columns(2)
    
    with col1:
        title = st.text_input("Track Title", value=f"Track {datetime.now().strftime('%H%M%S')}")
        genre = st.selectbox("Genre", ["pop", "rock", "jazz", "classical", "electronic"], key="music_genre")
        key = st.selectbox("Key", ["C", "D", "E", "F", "G", "A", "B"], key="music_key")
    
    with col2:
        mode = st.selectbox("Mode", ["major", "minor"], key="music_mode")
        tempo = st.slider("Tempo (BPM)", 60, 180, 120, key="music_tempo")
        num_bars = st.slider("Number of Bars", 4, 32, 8, key="music_bars")
    
    if st.button("🎼 Generate Music", use_container_width=True):
        with st.spinner("Generating music..."):
            generate_music_only(title, genre, key, mode, tempo, num_bars)


def render_lyrics_only_creator():
    """Render lyrics-only creation interface"""
    st.markdown("#### Generate Lyrics Only")
    
    col1, col2 = st.columns(2)
    
    with col1:
        title = st.text_input("Song Title", value=f"Lyrics {datetime.now().strftime('%H%M%S')}", key="lyrics_title")
        genre = st.selectbox("Genre", ["pop", "rock", "jazz", "classical"], key="lyrics_genre")
    
    with col2:
        emotion = st.selectbox("Emotion", ["happy", "sad", "energetic", "calm"], key="lyrics_emotion")
    
    if st.button("📝 Generate Lyrics", use_container_width=True):
        with st.spinner("Writing lyrics..."):
            generate_lyrics_only(title, genre, emotion)


def generate_full_song(title, genre, emotion, key, mode, tempo, num_bars, gen_lyrics, gen_art):
    """Generate a complete song with music, lyrics, and album art"""
    try:
        # Sanitize title to prevent path injection
        safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
        if not safe_title:
            safe_title = 'untitled'
        
        # Create output paths with sanitized title
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        music_filename = f'{timestamp}_{safe_title}.mid'
        art_filename = f'{timestamp}_{safe_title}.png'
        
        music_path = os.path.join('static', 'audio', music_filename)
        art_path = os.path.join('static', 'images', art_filename)
        
        # Generate music
        music_gen = MusicGenerator(genre=genre, key=key, mode=mode, emotion=emotion, tempo=tempo)
        music_gen.generate_midi(music_path, title)
        
        # Generate lyrics
        lyrics = None
        if gen_lyrics:
            lyric_gen = LyricGenerator(genre=genre, emotion=emotion)
            lyrics = lyric_gen.generate_full_lyrics()
        
        # Generate album art
        album_art_path = None
        if gen_art:
            art_gen = AlbumArtGenerator(emotion=emotion, genre=genre)
            album_art_path = art_gen.generate(title, art_path)
        
        # Calculate duration (rough estimate)
        duration = (num_bars * 4 * 60) / tempo  # bars * beats_per_bar * seconds_per_minute / tempo
        
        # Save to database
        song_id = SongService.create_song(
            title=title,
            artist="AI Generated",
            genre=genre,
            duration=duration,
            file_path=music_path,
            lyrics=lyrics,
            album_art_path=album_art_path,
            key=key,
            tempo=tempo
        )
        
        st.success(f"✅ Successfully generated '{title}'!")
        
        # Preview
        st.markdown("### Preview")
        col1, col2 = st.columns(2)
        
        with col1:
            if album_art_path and os.path.exists(album_art_path):
                st.image(album_art_path, caption="Album Art", width=300)
        
        with col2:
            st.markdown(f"**{title}**")
            st.markdown(f"Genre: {genre} | Key: {key} {mode}")
            st.markdown(f"Tempo: {tempo} BPM | Emotion: {emotion}")
            
            if os.path.exists(music_path):
                with open(music_path, 'rb') as f:
                    st.download_button(
                        "⬇️ Download MIDI",
                        f.read(),
                        file_name=os.path.basename(music_path)
                    )
        
        if lyrics:
            with st.expander("📝 View Generated Lyrics"):
                st.text(lyrics)
        
        # Set as current song
        st.session_state.current_song_id = song_id
        
    except Exception as e:
        st.error(f"Error generating song: {str(e)}")


def generate_music_only(title, genre, key, mode, tempo, num_bars):
    """Generate music only"""
    try:
        # Sanitize title to prevent path injection
        safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
        if not safe_title:
            safe_title = 'untitled'
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        music_filename = f'{timestamp}_{safe_title}.mid'
        music_path = os.path.join('static', 'audio', music_filename)
        
        music_gen = MusicGenerator(genre=genre, key=key, mode=mode, tempo=tempo)
        music_gen.generate_midi(music_path, title)
        
        duration = (num_bars * 4 * 60) / tempo
        
        song_id = SongService.create_song(
            title=title,
            artist="AI Generated",
            genre=genre,
            duration=duration,
            file_path=music_path,
            key=key,
            tempo=tempo
        )
        
        st.success(f"✅ Successfully generated music for '{title}'!")
        st.session_state.current_song_id = song_id
        
    except Exception as e:
        st.error(f"Error generating music: {str(e)}")


def generate_lyrics_only(title, genre, emotion):
    """Generate lyrics only"""
    try:
        lyric_gen = LyricGenerator(genre=genre, emotion=emotion)
        lyrics = lyric_gen.generate_full_lyrics()
        
        st.success(f"✅ Successfully generated lyrics for '{title}'!")
        
        st.markdown("### Generated Lyrics")
        st.text(lyrics)
        
        # Option to download
        st.download_button(
            "⬇️ Download Lyrics",
            lyrics,
            file_name=f"{title.replace(' ', '_')}_lyrics.txt"
        )
        
    except Exception as e:
        st.error(f"Error generating lyrics: {str(e)}")
