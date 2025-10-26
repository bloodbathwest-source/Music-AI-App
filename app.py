"""
Music AI App
A comprehensive music generation application with AI-powered features
"""
import streamlit as st
from modules import (
    MusicGenerator, 
    LyricGenerator, 
    MIDIHandler, 
    Visualizer,
    get_music_writers_table
)


def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="Music AI App",
        page_icon="🎵",
        layout="wide"
    )
    
    st.title("🎵 Music AI App - AI-Powered Music Generation")
    st.markdown("Generate original music with AI-powered melody, lyrics, and album art!")
    
    # Create tabs for different features
    tab1, tab2, tab3 = st.tabs(["🎼 Music Generator", "📝 Lyrics Generator", "ℹ️ About"])
    
    with tab1:
        music_generation_tab()
    
    with tab2:
        lyrics_generation_tab()
    
    with tab3:
        about_tab()


def music_generation_tab():
    """Music generation interface"""
    st.header("Generate AI Music")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Music Parameters")
        
        # User inputs for music generation
        genre = st.selectbox(
            "Genre",
            ["pop", "jazz", "classical", "rock"],
            help="Select the musical genre"
        )
        
        key_root = st.selectbox(
            "Key Root",
            ["C", "D", "E", "F", "G", "A", "B"],
            help="Select the root note of the key"
        )
        
        mode = st.selectbox(
            "Mode",
            ["major", "minor", "dorian", "phrygian", "lydian", "mixolydian", "locrian"],
            help="Select the musical mode/scale"
        )
        
        emotion = st.selectbox(
            "Emotion",
            ["happy", "sad", "suspenseful", "energetic"],
            help="Select the emotional tone"
        )
        
        tempo = st.slider(
            "Tempo (BPM)",
            60, 180, 120,
            help="Beats per minute"
        )
        
        generations = st.slider(
            "Evolution Generations",
            10, 100, 30,
            help="More generations = better quality but slower"
        )
        
        melody_length = st.slider(
            "Melody Length (notes)",
            16, 64, 32,
            help="Number of notes in the melody"
        )
        
        # Generate button
        if st.button("🎵 Generate Music", type="primary", use_container_width=True):
            generate_music(genre, key_root, mode, emotion, tempo, generations, melody_length)
    
    with col2:
        st.subheader("Generated Music")
        if "generated_music" not in st.session_state:
            st.info("👈 Configure parameters and click 'Generate Music' to start!")
            
            # Show music writers table
            st.subheader("Respected Music Writers & Composers")
            writers = get_music_writers_table()
            st.table(writers)
        else:
            display_generated_music()


def generate_music(genre, key_root, mode, emotion, tempo, generations, melody_length):
    """Generate music based on parameters"""
    with st.spinner("🎵 Evolving your music... This may take a moment..."):
        # Initialize generators
        music_gen = MusicGenerator(
            key_root=key_root,
            mode=mode,
            genre=genre,
            emotion=emotion
        )
        
        midi_handler = MIDIHandler(tempo=tempo)
        visualizer = Visualizer()
        
        # Generate melody using genetic algorithm
        best_individual = music_gen.evolve(
            generations=generations,
            population_size=20,
            melody_length=melody_length
        )
        
        # Create MIDI file
        midi_buffer = midi_handler.melody_to_midi(best_individual)
        
        # Get statistics
        stats = midi_handler.get_midi_stats(best_individual["melody"])
        
        # Create visualization
        melody_fig = visualizer.plot_melody(
            best_individual["melody"],
            title=f"{genre.title()} Melody in {key_root} {mode.title()}"
        )
        
        # Generate album art
        title = f"{emotion.title()} {genre.title()} in {key_root} {mode.title()}"
        album_art = visualizer.generate_album_art(
            title=title,
            artist="AI Music Generator",
            theme=genre
        )
        
        # Store in session state
        st.session_state.generated_music = {
            "midi_buffer": midi_buffer,
            "melody_fig": melody_fig,
            "album_art": album_art,
            "stats": stats,
            "params": {
                "genre": genre,
                "key": key_root,
                "mode": mode,
                "emotion": emotion,
                "tempo": tempo,
                "generations": generations,
                "melody_length": melody_length
            }
        }
    
    st.success("✅ Music generated successfully!")
    st.rerun()


def display_generated_music():
    """Display the generated music and related content"""
    music_data = st.session_state.generated_music
    
    # Display album art
    st.image(music_data["album_art"], use_container_width=True)
    
    # Display parameters
    params = music_data["params"]
    st.markdown(f"""
    **Generated Music Details:**
    - **Genre:** {params['genre'].title()}
    - **Key:** {params['key']} {params['mode'].title()}
    - **Emotion:** {params['emotion'].title()}
    - **Tempo:** {params['tempo']} BPM
    - **Generations:** {params['generations']}
    """)
    
    # Display statistics
    stats = music_data["stats"]
    st.markdown(f"""
    **Melody Statistics:**
    - **Total Notes:** {stats['total_notes']}
    - **Duration:** {stats['total_duration']:.1f} beats
    - **Note Range:** {stats['note_range']} semitones
    - **Unique Notes:** {stats['unique_notes']}
    """)
    
    # Display melody visualization
    st.pyplot(music_data["melody_fig"])
    
    # MIDI download
    st.download_button(
        label="📥 Download MIDI File",
        data=music_data["midi_buffer"].getvalue(),
        file_name="generated_music.mid",
        mime="audio/midi",
        use_container_width=True
    )
    
    # Note: Audio playback would require MIDI to WAV conversion
    # which needs additional dependencies like FluidSynth
    st.info("💡 Download the MIDI file and play it in your favorite music software!")
    
    # Button to generate new music
    if st.button("🔄 Generate New Music", use_container_width=True):
        del st.session_state.generated_music
        st.rerun()


def lyrics_generation_tab():
    """Lyrics generation interface"""
    st.header("Generate AI Lyrics")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Lyric Parameters")
        
        # User inputs for lyrics
        theme = st.selectbox(
            "Theme",
            ["love", "sadness", "adventure", "hope"],
            help="Select the lyrical theme"
        )
        
        style = st.selectbox(
            "Style",
            ["pop", "rock", "jazz", "classical"],
            help="Select the musical style"
        )
        
        emotion = st.selectbox(
            "Emotion",
            ["happy", "sad", "suspenseful", "hopeful"],
            help="Select the emotional tone",
            key="lyrics_emotion"
        )
        
        # Generate button
        if st.button("📝 Generate Lyrics", type="primary", use_container_width=True):
            generate_lyrics(theme, style, emotion)
    
    with col2:
        st.subheader("Generated Lyrics")
        if "generated_lyrics" not in st.session_state:
            st.info("👈 Configure parameters and click 'Generate Lyrics' to start!")
        else:
            display_generated_lyrics()


def generate_lyrics(theme, style, emotion):
    """Generate lyrics based on parameters"""
    with st.spinner("📝 Writing your lyrics..."):
        # Initialize lyric generator
        lyric_gen = LyricGenerator(theme=theme, style=style, emotion=emotion)
        
        # Generate lyrics
        lyrics_data = lyric_gen.generate_lyrics()
        formatted_lyrics = lyric_gen.format_lyrics(lyrics_data)
        
        # Generate album art for the lyrics
        visualizer = Visualizer()
        album_art = visualizer.generate_album_art(
            title=lyrics_data["title"],
            artist="AI Lyric Generator",
            theme=theme
        )
        
        # Store in session state
        st.session_state.generated_lyrics = {
            "lyrics": lyrics_data,
            "formatted": formatted_lyrics,
            "album_art": album_art,
            "params": {
                "theme": theme,
                "style": style,
                "emotion": emotion
            }
        }
    
    st.success("✅ Lyrics generated successfully!")
    st.rerun()


def display_generated_lyrics():
    """Display the generated lyrics"""
    lyrics_data = st.session_state.generated_lyrics
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Display album art
        st.image(lyrics_data["album_art"], use_container_width=True)
        
        # Display parameters
        params = lyrics_data["params"]
        st.markdown(f"""
        **Lyrics Details:**
        - **Theme:** {params['theme'].title()}
        - **Style:** {params['style'].title()}
        - **Emotion:** {params['emotion'].title()}
        """)
        
        # Button to generate new lyrics
        if st.button("🔄 Generate New Lyrics", use_container_width=True):
            del st.session_state.generated_lyrics
            st.rerun()
    
    with col2:
        # Display formatted lyrics
        st.markdown(lyrics_data["formatted"])
        
        # Download as text file
        st.download_button(
            label="📥 Download Lyrics",
            data=lyrics_data["formatted"],
            file_name=f"{lyrics_data['lyrics']['title']}.txt",
            mime="text/plain",
            use_container_width=True
        )


def about_tab():
    """About section with information"""
    st.header("About Music AI App")
    
    st.markdown("""
    ### 🎵 Welcome to Music AI App!
    
    This application uses artificial intelligence to generate original music and lyrics.
    Our AI combines multiple techniques to create unique musical compositions:
    
    #### Features:
    
    **Music Generation:**
    - 🧬 Genetic algorithm-based melody evolution
    - 🎼 Multiple musical scales and modes support
    - 🎹 Customizable tempo, key, and emotion
    - 📊 Visual melody analysis and statistics
    - 🎨 Auto-generated album art
    - 💾 MIDI file export
    
    **Lyrics Generation:**
    - ✍️ Theme-based lyric creation
    - 🎭 Multiple emotional tones
    - 📜 Structured songs (verses, chorus, bridge)
    - 🎨 Custom album art for each song
    - 💾 Text file export
    
    #### How It Works:
    
    1. **Music Generation**: Uses a genetic algorithm to evolve melodies over multiple 
       generations, optimizing for melodic contour, rhythm variety, and genre-specific 
       characteristics.
    
    2. **Lyrics Generation**: Employs template-based generation with theme-specific 
       word banks to create coherent and emotionally resonant lyrics.
    
    3. **Visualization**: Creates intuitive visual representations of the generated 
       music and custom album artwork.
    
    #### Respected Music Writers & Composers:
    """)
    
    # Display music writers table
    writers = get_music_writers_table()
    st.table(writers)
    
    st.markdown("""
    ---
    **Note:** This is an AI-powered tool designed for creative exploration and learning.
    The generated content is original and created by algorithms, not copied from existing works.
    """)


if __name__ == "__main__":
    main()