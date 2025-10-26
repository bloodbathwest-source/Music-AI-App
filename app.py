import io
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from midiutil import MIDIFile
from PIL import Image, ImageDraw

# Page configuration
st.set_page_config(
    page_title="Music AI App",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'library' not in st.session_state:
    st.session_state.library = []
if 'current_track' not in st.session_state:
    st.session_state.current_track = None
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'
if 'feedback' not in st.session_state:
    st.session_state.feedback = []

# Theme toggle
def toggle_theme():
    st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'

# Apply custom CSS based on theme
def apply_theme():
    if st.session_state.theme == 'dark':
        st.markdown("""
        <style>
        .main {background-color: #1e1e1e; color: #ffffff;}
        .stButton>button {background-color: #4CAF50; color: white;}
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .main {background-color: #ffffff; color: #000000;}
        .stButton>button {background-color: #2196F3; color: white;}
        </style>
        """, unsafe_allow_html=True)

apply_theme()

# Sidebar navigation
with st.sidebar:
    st.title("🎵 Music AI App")

    # Theme toggle button
    if st.button("🌓 Toggle Theme"):
        toggle_theme()
        st.rerun()

    st.markdown("---")
    navigation_options = [
        "🎹 AI Creator", "📚 Library", "🎧 Player", "🎨 Album Art",
        "✍️ Lyrics", "📊 Visualizations", "💬 Feedback"
    ]
    page = st.radio("Navigation", navigation_options)

    st.markdown("---")
    st.markdown("### Quick Stats")
    st.metric("Tracks Generated", len(st.session_state.library))
    st.metric("Feedback Received", len(st.session_state.feedback))

# Helper Functions
def generate_music_data(genre, key_root, mode, emotion, complexity):
    """Generate music based on parameters"""
    root_notes = {"C": 60, "D": 62, "E": 64, "F": 65, "G": 67, "A": 69, "B": 71}
    scales = {
        "major": [0, 2, 4, 5, 7, 9, 11],
        "minor": [0, 2, 3, 5, 7, 8, 10],
        "dorian": [0, 2, 3, 5, 7, 9, 10]
    }

    root = root_notes[key_root]
    scale = scales[mode]
    melody_notes = [root + i + 12 for i in scale]

    # Generate melody based on emotion
    num_notes = 32 if complexity == "Simple" else 64 if complexity == "Medium" else 96

    melody = []
    for _ in range(num_notes):
        if emotion == "happy":
            note = random.choice(melody_notes[3:])  # Higher notes
            duration = random.choice([0.5, 1.0])
        elif emotion == "sad":
            note = random.choice(melody_notes[:4])  # Lower notes
            duration = random.choice([1.0, 2.0])
        else:  # suspenseful
            note = random.choice(melody_notes)
            duration = random.choice([0.25, 0.5, 1.0])

        velocity = random.randint(60, 100)
        melody.append((note, duration, velocity))

    return melody

def create_midi_file(melody, tempo=120):
    """Create MIDI file from melody"""
    midi = MIDIFile(1)
    midi.addTempo(0, 0, tempo)

    time = 0
    for note, duration, velocity in melody:
        midi.addNote(0, 0, note, time, duration, velocity)
        time += duration

    midi_buffer = io.BytesIO()
    midi.writeFile(midi_buffer)
    midi_buffer.seek(0)
    return midi_buffer

def generate_lyrics(theme, mood, num_lines=8):
    """Generate simple lyrics based on theme and mood"""
    themes_words = {
        "love": ["heart", "love", "together", "forever", "feelings", "emotions"],
        "nature": ["sky", "trees", "mountains", "rivers", "earth", "wind"],
        "life": ["journey", "path", "dreams", "hopes", "time", "moments"],
        "adventure": ["explore", "discover", "journey", "wandering", "horizons", "quest"]
    }

    mood_templates = {
        "happy": [
            "Dancing through the {word1}",
            "Joy fills the {word2}",
            "Sunshine in my {word3}"
        ],
        "sad": [
            "Lonely {word1}",
            "Missing the {word2}",
            "Tears fall like {word3}"
        ],
        "energetic": [
            "Running wild through {word1}",
            "Energy fills the {word2}",
            "Never stop the {word3}"
        ]
    }

    words = themes_words.get(theme, themes_words["life"])
    templates = mood_templates.get(mood, mood_templates["happy"])

    lyrics = []
    for i in range(num_lines):
        template = random.choice(templates)
        line = template.format(
            word1=random.choice(words),
            word2=random.choice(words),
            word3=random.choice(words)
        )
        lyrics.append(line)

    return "\n".join(lyrics)

def create_album_art(title, artist, color_scheme):
    """Generate simple album art"""
    img_size = (400, 400)
    img = Image.new('RGB', img_size, color=color_scheme[0])
    draw = ImageDraw.Draw(img)

    # Draw geometric patterns
    for i in range(5):
        x = random.randint(0, 300)
        y = random.randint(0, 300)
        size = random.randint(50, 150)
        color = random.choice(color_scheme)
        draw.ellipse([x, y, x+size, y+size], fill=color, outline=color_scheme[1])

    # Add text (simplified - would need font file for production)
    draw.rectangle([50, 300, 350, 380], fill=(0, 0, 0, 128))

    return img

def create_waveform_viz(melody):
    """Create waveform visualization"""
    times = []
    notes = []
    current_time = 0

    for note, duration, velocity in melody:
        times.append(current_time)
        notes.append(note)
        current_time += duration

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times,
        y=notes,
        mode='lines+markers',
        name='Waveform',
        line={'color': '#4CAF50', 'width': 2},
        marker={'size': 6}
    ))

    fig.update_layout(
        title="Music Waveform Visualization",
        xaxis_title="Time (beats)",
        yaxis_title="Note Pitch (MIDI)",
        template="plotly_dark" if st.session_state.theme == 'dark' else "plotly_white",
        height=400
    )

    return fig

def create_piano_roll(melody):
    """Create piano roll visualization"""
    fig, ax = plt.subplots(figsize=(12, 6))

    current_time = 0
    for note, duration, velocity in melody:
        height = velocity / 100.0
        ax.barh(note, duration, left=current_time, height=height,
                color=plt.cm.viridis(velocity/127.0), alpha=0.8)
        current_time += duration

    ax.set_xlabel('Time (beats)')
    ax.set_ylabel('MIDI Note')
    ax.set_title('Piano Roll Visualization')
    ax.grid(True, alpha=0.3)

    return fig

# Main Pages
if page == "🎹 AI Creator":
    st.header("🎹 AI Music Creator")
    st.write("Generate unique music compositions using AI")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Musical Parameters")
        genre = st.selectbox("Genre", ["pop", "jazz", "classical", "rock", "electronic"])
        key_root = st.selectbox("Key", ["C", "D", "E", "F", "G", "A", "B"])
        mode = st.selectbox("Mode", ["major", "minor", "dorian"])

    with col2:
        st.subheader("Emotional Settings")
        emotion = st.selectbox("Emotion", ["happy", "sad", "suspenseful", "energetic"])
        complexity = st.selectbox("Complexity", ["Simple", "Medium", "Complex"])
        tempo = st.slider("Tempo (BPM)", 60, 180, 120)

    track_name = st.text_input("Track Name", f"{genre.title()} {emotion.title()} Melody")

    if st.button("🎵 Generate Music", type="primary"):
        with st.spinner("Creating your music..."):
            # Generate music
            melody = generate_music_data(genre, key_root, mode, emotion, complexity)
            midi_buffer = create_midi_file(melody, tempo)

            # Create track info
            track_info = {
                'name': track_name,
                'genre': genre,
                'key': key_root,
                'mode': mode,
                'emotion': emotion,
                'complexity': complexity,
                'tempo': tempo,
                'melody': melody,
                'created': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            # Add to library
            st.session_state.library.append(track_info)
            st.session_state.current_track = track_info

            st.success(f"✅ Created: {track_name}")

            # Display preview
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📥 Download MIDI",
                    data=midi_buffer,
                    file_name=f"{track_name}.mid",
                    mime="audio/midi"
                )

            with col2:
                st.info(f"Generated {len(melody)} notes")

            # Show visualization
            st.plotly_chart(create_waveform_viz(melody), use_container_width=True)

elif page == "📚 Library":
    st.header("📚 Music Library")
    st.write(f"You have {len(st.session_state.library)} track(s) in your library")

    # Search functionality
    search_query = st.text_input("🔍 Search tracks", "")

    if st.session_state.library:
        filtered_library = st.session_state.library
        if search_query:
            filtered_library = [
                track for track in st.session_state.library
                if search_query.lower() in track['name'].lower() or
                   search_query.lower() in track['genre'].lower()
            ]

        for idx, track in enumerate(filtered_library):
            with st.expander(f"🎵 {track['name']} - {track['genre'].title()}"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.write(f"**Key:** {track['key']} {track['mode']}")
                    st.write(f"**Emotion:** {track['emotion']}")

                with col2:
                    st.write(f"**Tempo:** {track['tempo']} BPM")
                    st.write(f"**Complexity:** {track['complexity']}")

                with col3:
                    st.write(f"**Created:** {track['created']}")
                    st.write(f"**Notes:** {len(track['melody'])}")

                if st.button("▶️ Play", key=f"play_{idx}"):
                    st.session_state.current_track = track
                    st.success(f"Now playing: {track['name']}")
    else:
        st.info("Your library is empty. Create some music in the AI Creator!")

elif page == "🎧 Player":
    st.header("🎧 Music Player")

    if st.session_state.current_track:
        track = st.session_state.current_track

        st.subheader(f"Now Playing: {track['name']}")

        col1, col2 = st.columns([2, 1])

        with col1:
            # Create and display MIDI
            midi_buffer = create_midi_file(track['melody'], track['tempo'])
            st.download_button(
                label="📥 Download MIDI",
                data=midi_buffer,
                file_name=f"{track['name']}.mid",
                mime="audio/midi"
            )

            # Visualizations
            st.plotly_chart(create_waveform_viz(track['melody']), use_container_width=True)

        with col2:
            st.markdown("### Track Info")
            st.write(f"**Genre:** {track['genre'].title()}")
            st.write(f"**Key:** {track['key']} {track['mode']}")
            st.write(f"**Emotion:** {track['emotion'].title()}")
            st.write(f"**Tempo:** {track['tempo']} BPM")
            st.write(f"**Duration:** {len(track['melody'])} notes")

        # Piano roll
        st.subheader("Piano Roll")
        piano_fig = create_piano_roll(track['melody'])
        st.pyplot(piano_fig)
        plt.close()

    else:
        st.info("No track selected. Choose a track from the Library or create one in AI Creator!")

elif page == "🎨 Album Art":
    st.header("🎨 Album Art Creator")
    st.write("Generate custom album artwork for your music")

    col1, col2 = st.columns(2)

    with col1:
        art_title = st.text_input("Album Title", "My Album")
        artist_name = st.text_input("Artist Name", "AI Composer")

        color_scheme_option = st.selectbox(
            "Color Scheme",
            ["Vibrant", "Pastel", "Dark", "Neon", "Earth Tones"]
        )

        color_schemes = {
            "Vibrant": [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)],
            "Pastel": [(255, 182, 193), (173, 216, 230), (255, 218, 185), (221, 160, 221)],
            "Dark": [(50, 50, 50), (100, 100, 100), (75, 0, 130), (139, 0, 139)],
            "Neon": [(255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0)],
            "Earth Tones": [(139, 69, 19), (160, 82, 45), (205, 133, 63), (222, 184, 135)]
        }

        selected_colors = color_schemes[color_scheme_option]

    with col2:
        st.write("Color Preview")
        preview_cols = st.columns(len(selected_colors))
        for i, color in enumerate(selected_colors):
            with preview_cols[i]:
                st.color_picker("", f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
                              key=f"color_{i}", disabled=True)

    if st.button("🎨 Generate Album Art", type="primary"):
        with st.spinner("Creating album art..."):
            album_art = create_album_art(art_title, artist_name, selected_colors)

            st.subheader("Generated Album Art")
            st.image(album_art, caption=f"{art_title} by {artist_name}", width=400)

            # Save to buffer for download
            img_buffer = io.BytesIO()
            album_art.save(img_buffer, format='PNG')
            img_buffer.seek(0)

            st.download_button(
                label="📥 Download Album Art",
                data=img_buffer,
                file_name=f"{art_title}_album_art.png",
                mime="image/png"
            )

elif page == "✍️ Lyrics":
    st.header("✍️ AI Lyric Generator")
    st.write("Generate song lyrics based on themes and moods")

    col1, col2 = st.columns(2)

    with col1:
        theme = st.selectbox("Theme", ["love", "nature", "life", "adventure"])
        mood = st.selectbox("Mood", ["happy", "sad", "energetic", "contemplative"])

    with col2:
        num_lines = st.slider("Number of Lines", 4, 16, 8)
        song_title = st.text_input("Song Title", f"{theme.title()} Song")

    if st.button("✍️ Generate Lyrics", type="primary"):
        with st.spinner("Writing lyrics..."):
            lyrics = generate_lyrics(theme, mood, num_lines)

            st.subheader(f"🎵 {song_title}")
            st.markdown("---")
            st.markdown(lyrics)
            st.markdown("---")

            st.download_button(
                label="📥 Download Lyrics",
                data=lyrics,
                file_name=f"{song_title}_lyrics.txt",
                mime="text/plain"
            )

elif page == "📊 Visualizations":
    st.header("📊 Music Visualizations")

    if st.session_state.library:
        # Select track to visualize
        track_names = [track['name'] for track in st.session_state.library]
        selected_track_name = st.selectbox("Select Track", track_names)

        # Find the selected track
        selected_track = next(
            (track for track in st.session_state.library if track['name'] == selected_track_name),
            None
        )

        if selected_track:
            st.subheader(f"Visualizing: {selected_track['name']}")

            tab1, tab2, tab3 = st.tabs(["Waveform", "Piano Roll", "Statistics"])

            with tab1:
                st.plotly_chart(create_waveform_viz(selected_track['melody']),
                              use_container_width=True)

            with tab2:
                piano_fig = create_piano_roll(selected_track['melody'])
                st.pyplot(piano_fig)
                plt.close()

            with tab3:
                melody = selected_track['melody']
                notes = [n for n, _, _ in melody]
                velocities = [v for _, _, v in melody]

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Total Notes", len(melody))
                    st.metric("Avg Note", f"{np.mean(notes):.1f}")

                with col2:
                    st.metric("Highest Note", max(notes))
                    st.metric("Lowest Note", min(notes))

                with col3:
                    st.metric("Avg Velocity", f"{np.mean(velocities):.1f}")
                    st.metric("Note Range", max(notes) - min(notes))

                # Histogram
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

                ax1.hist(notes, bins=20, color='#4CAF50', alpha=0.7)
                ax1.set_xlabel('MIDI Note')
                ax1.set_ylabel('Frequency')
                ax1.set_title('Note Distribution')
                ax1.grid(True, alpha=0.3)

                ax2.hist(velocities, bins=20, color='#2196F3', alpha=0.7)
                ax2.set_xlabel('Velocity')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Velocity Distribution')
                ax2.grid(True, alpha=0.3)

                st.pyplot(fig)
                plt.close()
    else:
        st.info("No tracks available. Create some music first!")

elif page == "💬 Feedback":
    st.header("💬 User Feedback System")
    st.write("Share your thoughts and help improve the AI!")

    # Feedback form
    with st.form("feedback_form"):
        st.subheader("Submit Feedback")

        if st.session_state.library:
            track_names = ["General Feedback"]
            track_names.extend([
                track['name'] for track in st.session_state.library
            ])
            feedback_track = st.selectbox("About Track", track_names)
        else:
            feedback_track = "General Feedback"

        rating = st.slider("Rating", 1, 5, 3)
        feedback_category = st.selectbox(
            "Category",
            ["Music Quality", "User Interface", "Features", "Performance", "Other"]
        )
        feedback_text = st.text_area("Your Feedback", height=150)

        submitted = st.form_submit_button("Submit Feedback")

        if submitted:
            feedback_entry = {
                'track': feedback_track,
                'rating': rating,
                'category': feedback_category,
                'feedback': feedback_text,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state.feedback.append(feedback_entry)
            st.success("Thank you for your feedback! 🎉")

    # Display feedback
    st.markdown("---")
    st.subheader("Recent Feedback")

    if st.session_state.feedback:
        for idx, fb in enumerate(reversed(st.session_state.feedback)):
            with st.expander(f"{'⭐' * fb['rating']} - {fb['category']} ({fb['timestamp']})"):
                st.write(f"**Track:** {fb['track']}")
                st.write(f"**Category:** {fb['category']}")
                st.write(f"**Rating:** {fb['rating']}/5")
                st.write(f"**Feedback:** {fb['feedback']}")
    else:
        st.info("No feedback yet. Be the first to share your thoughts!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🎵 Music AI App - Generate, Create, and Explore AI Music</p>
        <p>Built with Streamlit | Powered by AI</p>
    </div>
    """,
    unsafe_allow_html=True
)

if __name__ == "__main__":
    pass
