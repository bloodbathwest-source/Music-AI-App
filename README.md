# Music AI App 🎵

An AI-powered music creation application that generates original music based on user preferences and learns from feedback over time.

## Features

### 🎼 AI Music Generation
- Generate original melodies, harmonies, and rhythms
- Support for multiple genres: Pop, Jazz, Classical, Rock, Electronic, Ambient
- Emotional mood selection: Happy, Sad, Suspenseful, Energetic, Calm, Dramatic
- Musical key and mode/scale customization
- Evolutionary algorithm for music optimization

### 🧠 Self-Learning Mechanism
- **User Feedback Integration**: Rate generated music from 1-5 stars
- **Adaptive Learning**: AI improves based on user ratings
- **Pattern Recognition**: Learns successful combinations of genre, mood, and parameters
- **Continuous Improvement**: Each piece of feedback refines future generations

### 🎨 Interactive UI
- **Streamlit-based Interface**: Clean, intuitive web interface
- **Real-time Generation**: Watch music being created
- **Parameter Controls**: Fine-tune genre, mood, key, mode, and more
- **Advanced Settings**: Control evolution generations and melody length

### 📊 Visualizations
- **Waveform Display**: Interactive audio waveform visualization
- **Note Distribution**: Bar chart showing note frequency
- **Piano Roll Notation**: Visual representation of the musical score
- **Detailed Note Data**: View individual note information

### 💾 Download & Playback
- Download generated music as MIDI files
- View music statistics (tempo, duration, note count)
- Access complete music history

### 🔧 Backend API
- RESTful API built with FastAPI
- Endpoints for music generation, feedback submission, and statistics
- SQLite database for persistent storage
- Scalable architecture

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/bloodbathwest-source/Music-AI-App.git
cd Music-AI-App
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Initialize the database (automatic on first run):
The database will be created automatically when you first run the app.

## Usage

### Running the Streamlit App

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

### Running the Backend API (Optional)

For standalone API access:

```bash
python -m backend.api.main
```

The API will be available at `http://localhost:8000`

API documentation: `http://localhost:8000/docs`

## How to Use the App

1. **Select Parameters**:
   - Choose a genre (Pop, Jazz, Classical, etc.)
   - Select a mood (Happy, Sad, Energetic, etc.)
   - Pick a musical key (C, D, E, etc.)
   - Choose a mode/scale (Major, Minor, Pentatonic, etc.)
   - Optionally add a theme

2. **Advanced Settings**:
   - Adjust evolution generations (10-100)
   - Set melody length (16-64 notes)

3. **Generate Music**:
   - Click "Generate Music" button
   - Wait for the AI to evolve your music
   - View the generated piece

4. **Explore Visualizations**:
   - View the waveform
   - Analyze note distribution
   - Examine the piano roll notation

5. **Provide Feedback**:
   - Rate the music (1-5 stars)
   - Add optional comments
   - Submit to help the AI learn

6. **Download**:
   - Click "Download MIDI" to save the file
   - Use in your DAW or music software

## Self-Learning Mechanism

The AI music generator uses a sophisticated self-learning approach:

### How It Works

1. **Initial Generation**: Uses genetic algorithm with predefined patterns
2. **User Feedback Collection**: Stores ratings and comments in database
3. **Pattern Analysis**: Analyzes which combinations work best
4. **Parameter Adjustment**: Adjusts generation parameters based on feedback
5. **Continuous Improvement**: Each generation is better than the last

### Technical Details

- **Genetic Algorithm**: Evolves music through selection, crossover, and mutation
- **LSTM Neural Network**: Planned for sequence prediction (foundation laid)
- **Feedback Loop**: avg_rating influences complexity and variation parameters
- **Training Data**: Stores successful patterns for reuse

### Database Schema

- **generated_music**: Stores all generated pieces
- **user_feedback**: Stores ratings and comments
- **model_training_data**: Aggregates feedback for learning

## Architecture

```
Music-AI-App/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── backend/
│   ├── models/
│   │   ├── music_generator.py      # AI music generation logic
│   │   └── music_utils.py          # MIDI and audio utilities
│   ├── database/
│   │   └── models.py               # SQLAlchemy database models
│   └── api/
│       └── main.py                 # FastAPI backend
├── generated_music/                # Output directory for MIDI files
└── music_ai.db                     # SQLite database (auto-created)
```

## API Endpoints

### POST /api/generate
Generate music based on parameters.

**Request Body:**
```json
{
  "genre": "pop",
  "mood": "happy",
  "key_root": "C",
  "mode": "major",
  "theme": "sunset",
  "generations": 20
}
```

### POST /api/feedback
Submit feedback for generated music.

**Request Body:**
```json
{
  "music_id": 1,
  "rating": 5,
  "comment": "Great melody!"
}
```

### GET /api/music/{music_id}
Get information about a specific music piece.

### GET /api/stats
Get overall statistics.

## Technologies Used

- **Streamlit**: Web interface
- **PyTorch**: Neural network framework
- **FastAPI**: Backend API
- **SQLAlchemy**: Database ORM
- **MIDIUtil**: MIDI file generation
- **NumPy/SciPy**: Numerical computations
- **Plotly/Matplotlib**: Visualizations
- **Librosa**: Audio processing

## Future Enhancements

- [ ] Integration with Hugging Face MusicGen for advanced generation
- [ ] WAV/MP3 audio file export
- [ ] Real-time audio playback in browser
- [ ] User accounts and music history
- [ ] Collaborative filtering for recommendations
- [ ] LSTM-based sequence prediction training
- [ ] Multi-track composition
- [ ] Export to music notation software

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with Streamlit for rapid web development
- Music theory implementation inspired by classical composition techniques
- Genetic algorithm approach for creative AI generation
