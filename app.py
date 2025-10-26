"""Music AI App - Main Streamlit Application."""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime
import os
import sys

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.models.music_generator import MusicGenerator
from backend.models.music_utils import MusicFileGenerator
from backend.database.models import (
    init_db, SessionLocal, GeneratedMusic, UserFeedback, ModelTrainingData
)

# Page configuration
st.set_page_config(
    page_title="Music AI App",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database
init_db()

# Initialize session state
if 'generated_music' not in st.session_state:
    st.session_state.generated_music = None
if 'music_id' not in st.session_state:
    st.session_state.music_id = None
if 'waveform_data' not in st.session_state:
    st.session_state.waveform_data = None


def main():
    """Main application function."""
    st.title("🎵 AI-Powered Music Creation Studio")
    st.markdown("""
    Create original music using AI with self-learning capabilities. 
    Generate melodies, provide feedback, and watch the AI improve over time!
    """)
    
    # Sidebar for user inputs
    with st.sidebar:
        st.header("🎹 Music Parameters")
        
        # Genre selection
        genre = st.selectbox(
            "Genre",
            ["pop", "jazz", "classical", "rock", "electronic", "ambient"],
            help="Choose the musical genre for your composition"
        )
        
        # Mood selection
        mood = st.selectbox(
            "Mood",
            ["happy", "sad", "suspenseful", "energetic", "calm", "dramatic"],
            help="Select the emotional tone of the music"
        )
        
        # Key selection
        key_root = st.selectbox(
            "Key",
            ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"],
            help="Choose the musical key"
        )
        
        # Mode selection
        mode = st.selectbox(
            "Mode/Scale",
            ["major", "minor", "dorian", "phrygian", "lydian", "mixolydian", "pentatonic"],
            help="Select the musical scale/mode"
        )
        
        # Theme (optional)
        theme = st.text_input(
            "Theme (optional)",
            placeholder="e.g., sunset, adventure, love",
            help="Optional theme to inspire the generation"
        )
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            generations = st.slider(
                "Evolution Generations",
                min_value=10,
                max_value=100,
                value=20,
                step=5,
                help="Number of evolutionary iterations (more = better quality, slower)"
            )
            
            melody_length = st.slider(
                "Melody Length (notes)",
                min_value=16,
                max_value=64,
                value=32,
                step=8,
                help="Length of the generated melody"
            )
        
        st.markdown("---")
        
        # Generate button
        generate_button = st.button(
            "🎼 Generate Music",
            type="primary",
            use_container_width=True
        )
    
    # Main content area
    if generate_button:
        generate_music(genre, mood, key_root, mode, theme, generations, melody_length)
    
    # Display generated music if available
    if st.session_state.generated_music is not None:
        display_music_player()
        display_visualizations()
        display_feedback_section()
    
    # Statistics section
    display_statistics()


def generate_music(genre, mood, key_root, mode, theme, generations, melody_length):
    """Generate music based on parameters."""
    with st.spinner("🎨 Creating your music... This may take a moment."):
        try:
            # Get database session
            db = SessionLocal()
            
            # Get feedback data for self-learning
            training_data = db.query(ModelTrainingData).filter(
                ModelTrainingData.genre == genre,
                ModelTrainingData.mood == mood
            ).all()
            
            feedback_data = [
                {
                    'genre': td.genre,
                    'mood': td.mood,
                    'parameters': td.parameters,
                    'avg_rating': td.avg_rating
                }
                for td in training_data
            ]
            
            # Initialize generator with feedback
            generator = MusicGenerator(feedback_data=feedback_data)
            
            # Generate music
            music_data = generator.evolve_music(
                genre=genre,
                key_root=key_root,
                mode=mode,
                mood=mood,
                generations=generations
            )
            
            # Create file generator
            file_gen = MusicFileGenerator()
            
            # Create MIDI file
            midi_path = file_gen.create_midi_file(music_data)
            
            # Generate waveform data
            waveform, sample_rate = file_gen.generate_waveform_data(music_data)
            
            # Calculate duration
            duration = sum(dur for _, dur, _ in music_data['melody'])
            
            # Save to database
            generated_music = GeneratedMusic(
                genre=genre,
                mood=mood,
                theme=theme or "",
                key_root=key_root,
                mode=mode,
                tempo=music_data['tempo'],
                duration=duration,
                file_path=midi_path
            )
            db.add(generated_music)
            db.commit()
            db.refresh(generated_music)
            
            # Store in session state
            st.session_state.generated_music = music_data
            st.session_state.music_id = generated_music.id
            st.session_state.waveform_data = (waveform, sample_rate)
            st.session_state.midi_path = midi_path
            
            db.close()
            
            st.success("✅ Music generated successfully!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error generating music: {str(e)}")


def display_music_player():
    """Display music player section."""
    st.header("🎧 Your Generated Music")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        metadata = st.session_state.generated_music.get('metadata', {})
        st.subheader(f"{metadata.get('genre', 'Unknown').title()} - {metadata.get('mood', 'Unknown').title()}")
        st.write(f"**Key:** {metadata.get('key', 'C')} {metadata.get('mode', 'major').title()}")
        st.write(f"**Tempo:** {st.session_state.generated_music.get('tempo', 120)} BPM")
    
    with col2:
        # Download MIDI button
        if os.path.exists(st.session_state.midi_path):
            with open(st.session_state.midi_path, 'rb') as f:
                midi_bytes = f.read()
            st.download_button(
                label="📥 Download MIDI",
                data=midi_bytes,
                file_name=f"music_{st.session_state.music_id}.mid",
                mime="audio/midi",
                use_container_width=True
            )
    
    with col3:
        # Info about the piece
        melody_notes = len(st.session_state.generated_music['melody'])
        st.metric("Notes", melody_notes)
    
    st.markdown("---")


def display_visualizations():
    """Display music visualizations."""
    st.header("📊 Visualizations")
    
    tab1, tab2, tab3 = st.tabs(["Waveform", "Note Distribution", "Music Notation"])
    
    with tab1:
        display_waveform()
    
    with tab2:
        display_note_distribution()
    
    with tab3:
        display_music_notation()


def display_waveform():
    """Display waveform visualization."""
    if st.session_state.waveform_data is not None:
        waveform, sample_rate = st.session_state.waveform_data
        
        # Create time array
        duration = len(waveform) / sample_rate
        time = np.linspace(0, duration, len(waveform))
        
        # Create interactive plot with Plotly
        fig = go.Figure()
        
        # Downsample for performance if needed
        if len(time) > 10000:
            step = len(time) // 10000
            time = time[::step]
            waveform = waveform[::step]
        
        fig.add_trace(go.Scatter(
            x=time,
            y=waveform,
            mode='lines',
            name='Waveform',
            line=dict(color='#1f77b4', width=1)
        ))
        
        fig.update_layout(
            title="Audio Waveform",
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)


def display_note_distribution():
    """Display note distribution chart."""
    melody = st.session_state.generated_music['melody']
    
    # Count note occurrences
    note_counts = {}
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    for note, _, _ in melody:
        note_name = note_names[note % 12]
        note_counts[note_name] = note_counts.get(note_name, 0) + 1
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=list(note_counts.keys()),
            y=list(note_counts.values()),
            marker_color='#2ca02c'
        )
    ])
    
    fig.update_layout(
        title="Note Distribution",
        xaxis_title="Note",
        yaxis_title="Count",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_music_notation():
    """Display simplified music notation."""
    melody = st.session_state.generated_music['melody']
    
    # Create piano roll visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    time = 0
    for note, duration, velocity in melody[:50]:  # Limit to first 50 notes
        # Normalize velocity for color
        color_intensity = velocity / 127.0
        
        # Plot note as rectangle
        ax.barh(
            note,
            duration,
            left=time,
            height=0.8,
            color=plt.cm.viridis(color_intensity),
            edgecolor='black',
            linewidth=0.5
        )
        time += duration
    
    ax.set_xlabel('Time (beats)', fontsize=12)
    ax.set_ylabel('MIDI Note Number', fontsize=12)
    ax.set_title('Piano Roll Notation', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Display note table
    with st.expander("📝 View Detailed Note Data"):
        file_gen = MusicFileGenerator()
        notation_data = file_gen.get_music_notation_data(st.session_state.generated_music)
        
        # Show first 20 notes
        st.write("First 20 notes:")
        for i, note_info in enumerate(notation_data[:20], 1):
            st.write(f"{i}. {note_info['note']} - Duration: {note_info['duration']}, Velocity: {note_info['velocity']}")


def display_feedback_section():
    """Display feedback submission section."""
    st.header("💭 Your Feedback")
    st.markdown("Help the AI improve by rating this music!")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        rating = st.slider(
            "Rating",
            min_value=1,
            max_value=5,
            value=3,
            help="Rate the music from 1 (poor) to 5 (excellent)"
        )
        
        # Star display
        stars = "⭐" * rating + "☆" * (5 - rating)
        st.markdown(f"### {stars}")
    
    with col2:
        comment = st.text_area(
            "Comments (optional)",
            placeholder="What did you like or dislike about this music?",
            height=100
        )
    
    if st.button("Submit Feedback", type="primary"):
        submit_feedback(rating, comment)


def submit_feedback(rating, comment):
    """Submit user feedback."""
    try:
        db = SessionLocal()
        
        # Save feedback
        feedback = UserFeedback(
            music_id=st.session_state.music_id,
            rating=rating,
            comment=comment
        )
        db.add(feedback)
        
        # Update or create training data
        music = db.query(GeneratedMusic).filter(
            GeneratedMusic.id == st.session_state.music_id
        ).first()
        
        training_data = db.query(ModelTrainingData).filter(
            ModelTrainingData.music_id == st.session_state.music_id
        ).first()
        
        if training_data:
            # Update existing
            old_count = training_data.feedback_count
            old_avg = training_data.avg_rating or 0
            new_count = old_count + 1
            new_avg = (old_avg * old_count + rating) / new_count
            
            training_data.avg_rating = new_avg
            training_data.feedback_count = new_count
        else:
            # Create new
            training_data = ModelTrainingData(
                music_id=st.session_state.music_id,
                genre=music.genre,
                mood=music.mood,
                parameters='{}',
                avg_rating=float(rating),
                feedback_count=1
            )
            db.add(training_data)
        
        db.commit()
        db.close()
        
        st.success("✅ Thank you for your feedback! The AI will learn from this.")
        
    except Exception as e:
        st.error(f"❌ Error submitting feedback: {str(e)}")


def display_statistics():
    """Display overall statistics."""
    st.markdown("---")
    st.header("📈 Statistics")
    
    try:
        db = SessionLocal()
        
        total_music = db.query(GeneratedMusic).count()
        total_feedback = db.query(UserFeedback).count()
        
        avg_ratings = db.query(UserFeedback.rating).all()
        avg_rating = sum(r[0] for r in avg_ratings) / len(avg_ratings) if avg_ratings else 0
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Music Generated", total_music)
        
        with col2:
            st.metric("Total Feedback Received", total_feedback)
        
        with col3:
            st.metric("Average Rating", f"{avg_rating:.2f} ⭐")
        
        db.close()
        
    except Exception as e:
        st.error(f"Error loading statistics: {str(e)}")


if __name__ == "__main__":
    main()