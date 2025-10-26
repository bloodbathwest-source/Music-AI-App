"""
Visualization component for waveforms and music notation
"""
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


def render_visualizations(current_song: Optional[dict] = None):
    """Render music visualizations"""
    st.markdown("### 📊 Visualizations")
    
    if not current_song:
        st.info("Select a song to view visualizations")
        return
    
    # Tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["🌊 Waveform", "🎼 Notation", "📈 Analysis"])
    
    with tab1:
        render_waveform_viz(current_song)
    
    with tab2:
        render_notation_viz(current_song)
    
    with tab3:
        render_analysis_viz(current_song)


def render_waveform_viz(song: dict):
    """Render waveform visualization"""
    st.markdown("#### Waveform Visualization")
    
    # Generate synthetic waveform for demonstration
    duration = song.get('duration', 30)
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create a simple waveform based on song parameters
    tempo = song.get('tempo', 120)
    frequency = 440  # A4 note
    
    # Generate wave with some variation
    wave = np.sin(2 * np.pi * frequency * t) * np.exp(-t / (duration / 2))
    
    # Add some harmonics
    wave += 0.5 * np.sin(2 * np.pi * frequency * 2 * t) * np.exp(-t / (duration / 2))
    wave += 0.3 * np.sin(2 * np.pi * frequency * 3 * t) * np.exp(-t / (duration / 2))
    
    # Downsample for visualization
    downsample_factor = 100
    t_display = t[::downsample_factor]
    wave_display = wave[::downsample_factor]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_display, wave_display, color='#1E88E5', linewidth=0.5)
    ax.fill_between(t_display, wave_display, alpha=0.3, color='#1E88E5')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Waveform - {song.get("title", "Unknown")}')
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Waveform stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Duration", f"{duration:.1f}s")
    with col2:
        st.metric("Peak Amplitude", f"{np.max(np.abs(wave)):.2f}")
    with col3:
        st.metric("RMS Level", f"{np.sqrt(np.mean(wave**2)):.2f}")


def render_notation_viz(song: dict):
    """Render music notation visualization"""
    st.markdown("#### Music Notation")
    
    # Generate simple notation display
    key = song.get('key', 'C')
    mode = song.get('mode', 'major')
    tempo = song.get('tempo', 120)
    
    st.markdown(f"""
    **Musical Parameters:**
    - **Key:** {key} {mode}
    - **Tempo:** {tempo} BPM
    - **Time Signature:** 4/4
    """)
    
    # Display note distribution
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Generate random note distribution for visualization
    note_counts = np.random.randint(5, 30, len(notes))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(notes, note_counts, color='#4CAF50', alpha=0.7)
    
    # Highlight tonic note
    tonic_index = notes.index(key) if key in notes else 0
    bars[tonic_index].set_color('#FF5722')
    
    ax.set_xlabel('Notes')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Note Distribution - Key of {key}')
    ax.grid(True, alpha=0.3, axis='y')
    
    st.pyplot(fig)
    
    st.info("💡 Red bar indicates the tonic (root) note of the key")


def render_analysis_viz(song: dict):
    """Render musical analysis"""
    st.markdown("#### Musical Analysis")
    
    genre = song.get('genre', 'Unknown')
    key = song.get('key', 'C')
    tempo = song.get('tempo', 120)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Genre characteristics
        st.markdown("**Genre Characteristics**")
        genre_features = {
            'pop': {'energy': 0.7, 'danceability': 0.8, 'acousticness': 0.3},
            'rock': {'energy': 0.9, 'danceability': 0.6, 'acousticness': 0.2},
            'jazz': {'energy': 0.5, 'danceability': 0.5, 'acousticness': 0.6},
            'classical': {'energy': 0.4, 'danceability': 0.3, 'acousticness': 0.9},
            'electronic': {'energy': 0.9, 'danceability': 0.9, 'acousticness': 0.1},
        }
        
        features = genre_features.get(genre.lower(), {'energy': 0.5, 'danceability': 0.5, 'acousticness': 0.5})
        
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
        
        categories = list(features.keys())
        values = list(features.values())
        
        # Complete the circle
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, color='#9C27B0')
        ax.fill(angles, values, alpha=0.25, color='#9C27B0')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Genre Profile', pad=20)
        ax.grid(True)
        
        st.pyplot(fig)
    
    with col2:
        # Tempo and key analysis
        st.markdown("**Musical Properties**")
        
        # Tempo category
        if tempo < 80:
            tempo_cat = "Slow (Adagio)"
        elif tempo < 120:
            tempo_cat = "Moderate (Andante)"
        elif tempo < 160:
            tempo_cat = "Fast (Allegro)"
        else:
            tempo_cat = "Very Fast (Presto)"
        
        st.metric("Tempo Category", tempo_cat)
        st.metric("BPM", tempo)
        
        # Key mood
        major_keys = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
        if key in major_keys:
            key_mood = "Bright/Happy"
        else:
            key_mood = "Dark/Moody"
        
        st.metric("Key Mood", key_mood)
        
        # Duration visualization
        duration = song.get('duration', 0)
        st.metric("Duration", f"{int(duration // 60)}m {int(duration % 60)}s")
        
        # Progress bars for features
        st.markdown("**Audio Features**")
        
        for feature, value in features.items():
            st.progress(value, text=f"{feature.capitalize()}: {int(value*100)}%")
