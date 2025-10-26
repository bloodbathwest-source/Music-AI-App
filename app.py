"""
Music AI App - A comprehensive music generation and management application
"""
import streamlit as st
from database import init_database, SongService, UserPreferenceService
from components import (
    render_player,
    render_library,
    render_search,
    render_ai_creator,
    render_visualizations
)
import os


# Page configuration
st.set_page_config(
    page_title="Music AI App",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)


def init_app():
    """Initialize the application"""
    # Initialize database
    init_database()
    
    # Initialize session state
    if 'current_song_id' not in st.session_state:
        st.session_state.current_song_id = None
    
    if 'theme' not in st.session_state:
        prefs = UserPreferenceService.get_preferences()
        st.session_state.theme = prefs.theme
    
    # Create static directories if they don't exist
    os.makedirs('static/audio', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)


def apply_custom_css():
    """Apply custom CSS styling"""
    theme = st.session_state.get('theme', 'dark')
    
    if theme == 'dark':
        primary_color = "#1E88E5"
        background_color = "#0E1117"
        secondary_bg = "#1E1E1E"
    else:
        primary_color = "#1976D2"
        background_color = "#FFFFFF"
        secondary_bg = "#F5F5F5"
    
    st.markdown(f"""
    <style>
        .main {{
            background-color: {background_color};
        }}
        .stButton>button {{
            background-color: {primary_color};
            color: white;
            border-radius: 5px;
            border: none;
            padding: 0.5rem 1rem;
            transition: all 0.3s;
        }}
        .stButton>button:hover {{
            background-color: {primary_color};
            opacity: 0.8;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        .song-card {{
            background-color: {secondary_bg};
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
        }}
        h1, h2, h3 {{
            color: {primary_color};
        }}
        .stProgress > div > div {{
            background-color: {primary_color};
        }}
    </style>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with navigation and settings"""
    with st.sidebar:
        st.title("🎵 Music AI App")
        st.markdown("---")
        
        # Navigation
        st.markdown("### Navigation")
        page = st.radio(
            "Go to",
            ["🏠 Home", "🎵 Library", "🔍 Search", "🤖 AI Creator", "📊 Visualizations"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Theme toggle
        st.markdown("### ⚙️ Settings")
        
        current_theme = st.session_state.get('theme', 'dark')
        theme_label = "🌙 Dark Mode" if current_theme == 'dark' else "☀️ Light Mode"
        
        if st.button(f"Toggle Theme ({theme_label})"):
            new_theme = 'light' if current_theme == 'dark' else 'dark'
            st.session_state.theme = new_theme
            UserPreferenceService.update_theme(new_theme)
            st.rerun()
        
        # App stats
        st.markdown("---")
        st.markdown("### 📊 Stats")
        
        songs = SongService.get_all_songs()
        st.metric("Total Songs", len(songs))
        
        if songs:
            genres = set(s.genre for s in songs if s.genre)
            st.metric("Genres", len(genres))
        
        # About
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.caption("Music AI App v1.0")
        st.caption("Generate, manage, and enjoy AI-created music")
        
        return page


def get_current_song():
    """Get current song data"""
    if st.session_state.current_song_id:
        song = SongService.get_song_by_id(st.session_state.current_song_id)
        if song:
            return {
                'id': song.id,
                'title': song.title,
                'artist': song.artist,
                'genre': song.genre,
                'duration': song.duration,
                'file_path': song.file_path,
                'lyrics': song.lyrics,
                'album_art_path': song.album_art_path,
                'key': song.key,
                'tempo': song.tempo,
                'mode': 'major'  # Default mode, could be stored in DB
            }
    return None


def main():
    """Main application function"""
    # Initialize app
    init_app()
    
    # Apply custom CSS
    apply_custom_css()
    
    # Render sidebar and get selected page
    page = render_sidebar()
    
    # Get current song
    current_song = get_current_song()
    
    # Render main content based on selected page
    if page == "🏠 Home":
        render_home_page(current_song)
    elif page == "🎵 Library":
        render_library()
    elif page == "🔍 Search":
        render_search()
    elif page == "🤖 AI Creator":
        render_ai_creator()
    elif page == "📊 Visualizations":
        render_visualizations(current_song)


def render_home_page(current_song):
    """Render the home page"""
    st.title("🎵 Welcome to Music AI App")
    st.markdown("""
    ### Create, Manage, and Enjoy AI-Generated Music
    
    This app provides a complete music creation and management experience powered by AI:
    
    - 🤖 **AI Music Generation**: Create original music with customizable parameters
    - 📝 **Lyric Generation**: Generate creative lyrics for any genre and emotion
    - 🎨 **Album Art**: Auto-generate beautiful album covers
    - 📚 **Music Library**: Organize and manage your creations
    - 🔍 **Smart Search**: Find songs quickly by title, artist, or genre
    - 📊 **Visualizations**: View waveforms, notation, and analysis
    
    **Get started by:**
    1. Navigate to "🤖 AI Creator" to generate your first song
    2. View your creations in "🎵 Library"
    3. Use "🔍 Search" to find specific songs
    """)
    
    st.markdown("---")
    
    # Display player if song is selected
    if current_song:
        render_player(current_song)
    else:
        # Quick start section
        st.markdown("### 🚀 Quick Start")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 1️⃣ Generate Music")
            st.markdown("Click on **AI Creator** to create your first song with custom parameters")
        
        with col2:
            st.markdown("#### 2️⃣ Manage Library")
            st.markdown("View and organize your music collection in the **Library**")
        
        with col3:
            st.markdown("#### 3️⃣ Explore & Enjoy")
            st.markdown("Use **Visualizations** to analyze and enjoy your creations")
        
        st.markdown("---")
        
        # Recent songs
        st.markdown("### 🎵 Recent Creations")
        songs = SongService.get_all_songs()
        
        if songs:
            recent_songs = songs[:6]  # Show last 6 songs
            
            cols = st.columns(3)
            for idx, song in enumerate(recent_songs):
                with cols[idx % 3]:
                    with st.container():
                        st.markdown(f"**{song.title}**")
                        st.caption(f"{song.artist} • {song.genre or 'N/A'}")
                        
                        if st.button("▶️ Play", key=f"home_play_{song.id}"):
                            st.session_state.current_song_id = song.id
                            st.rerun()
        else:
            st.info("No songs yet. Create your first song in the AI Creator!")


if __name__ == "__main__":
    main()