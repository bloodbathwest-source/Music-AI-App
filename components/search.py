"""
Search component for filtering and finding music
"""
import streamlit as st
from database.crud import SongService


def render_search():
    """Render the search component"""
    st.markdown("### 🔍 Search Music")
    
    # Search input
    search_query = st.text_input(
        "Search by title, artist, or genre",
        placeholder="Enter search terms...",
        key="search_input"
    )
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        filter_genre = st.selectbox(
            "Filter by Genre",
            ["All", "pop", "rock", "jazz", "classical", "electronic"],
            key="filter_genre"
        )
    
    with col2:
        filter_key = st.selectbox(
            "Filter by Key",
            ["All", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"],
            key="filter_key"
        )
    
    with col3:
        sort_order = st.selectbox(
            "Sort by",
            ["Recent", "Title (A-Z)", "Title (Z-A)", "Artist"],
            key="sort_order"
        )
    
    # Search button
    if st.button("🔍 Search", use_container_width=True):
        st.session_state.perform_search = True
    
    # Display results
    if search_query or st.session_state.get('perform_search', False):
        st.markdown("---")
        display_search_results(search_query, filter_genre, filter_key, sort_order)


def display_search_results(query: str, genre_filter: str, key_filter: str, sort_order: str):
    """Display search results"""
    # Get songs based on search
    if query:
        songs = SongService.search_songs(query)
    else:
        songs = SongService.get_all_songs()
    
    # Apply filters
    if genre_filter != "All":
        songs = [s for s in songs if s.genre == genre_filter]
    
    if key_filter != "All":
        songs = [s for s in songs if s.key == key_filter]
    
    # Apply sorting
    if sort_order == "Title (A-Z)":
        songs = sorted(songs, key=lambda x: x.title.lower())
    elif sort_order == "Title (Z-A)":
        songs = sorted(songs, key=lambda x: x.title.lower(), reverse=True)
    elif sort_order == "Artist":
        songs = sorted(songs, key=lambda x: x.artist.lower())
    else:  # Recent
        songs = sorted(songs, key=lambda x: x.created_at, reverse=True)
    
    # Display results
    st.markdown(f"### Found {len(songs)} song(s)")
    
    if songs:
        for song in songs:
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                
                with col1:
                    st.markdown(f"**{song.title}**")
                
                with col2:
                    st.text(song.artist)
                
                with col3:
                    st.text(f"{song.genre or 'N/A'} | {song.key or 'N/A'}")
                
                with col4:
                    if st.button("▶️ Play", key=f"search_play_{song.id}"):
                        st.session_state.current_song_id = song.id
                        st.rerun()
                
                st.markdown("---")
    else:
        st.info("No songs found matching your search criteria.")
