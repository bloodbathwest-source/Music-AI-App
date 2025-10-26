# Music AI App 🎵

A comprehensive AI-powered music creation application built with Streamlit. Generate music, lyrics, and album art using artificial intelligence.

## Features

### 🎹 AI Music Creator
- Generate unique music compositions based on:
  - Genre (pop, jazz, classical, rock, electronic)
  - Key and mode
  - Emotion (happy, sad, suspenseful, energetic)
  - Complexity levels
  - Custom tempo (BPM)
- Download generated music as MIDI files

### ✍️ Lyric Generation
- AI-powered lyric generation based on:
  - Themes (love, nature, life, adventure)
  - Moods (happy, sad, energetic, contemplative)
  - Customizable line count
- Download lyrics as text files

### 🎨 Album Art Creator
- Generate custom album artwork
- Multiple color schemes:
  - Vibrant, Pastel, Dark, Neon, Earth Tones
- Export as PNG images

### 🎧 Music Player
- Play generated music tracks
- View track information and metadata
- Real-time waveform visualization
- Piano roll display

### 📚 Library Management
- Store all generated tracks
- Search functionality
- Track metadata and creation dates

### 📊 Visualizations
- Interactive waveform displays
- Piano roll notation
- Statistical analysis of compositions
- Note and velocity distribution graphs

### 💬 Feedback System
- Submit feedback on generated music
- Rate tracks and features
- Help improve AI outputs

### 🌓 Theme Support
- Dark mode (default)
- Light mode
- Toggle between themes seamlessly

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

## Running Locally

Start the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## Deployment

### Streamlit Cloud (Recommended)

1. Fork or push this repository to GitHub
2. Visit [Streamlit Cloud](https://streamlit.io/cloud)
3. Sign in with your GitHub account
4. Click "New app"
5. Select your repository and branch
6. Set main file path to `app.py`
7. Click "Deploy"

Your app will be live at: `https://[your-app-name].streamlit.app`

### Other Deployment Options

#### Heroku
1. Create a `Procfile`:
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

2. Deploy:
```bash
heroku create your-app-name
git push heroku main
```

#### AWS EC2
1. Launch an EC2 instance
2. Install dependencies
3. Run with nohup:
```bash
nohup streamlit run app.py &
```

## Usage Guide

### Creating Music

1. Navigate to **AI Creator** from the sidebar
2. Select musical parameters:
   - Choose genre, key, and mode
   - Set emotion and complexity
   - Adjust tempo
3. Enter a track name
4. Click **Generate Music**
5. Download MIDI file or add to library

### Generating Lyrics

1. Go to **Lyrics** section
2. Select theme and mood
3. Choose number of lines
4. Click **Generate Lyrics**
5. Download or copy lyrics

### Creating Album Art

1. Open **Album Art** section
2. Enter album title and artist name
3. Select color scheme
4. Click **Generate Album Art**
5. Download PNG image

### Using the Player

1. Navigate to **Player**
2. Select a track from the library
3. View visualizations and track info
4. Download MIDI file

### Providing Feedback

1. Go to **Feedback** section
2. Select track or general feedback
3. Rate and categorize your feedback
4. Submit to help improve the AI

## Project Structure

```
Music-AI-App/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── .streamlit/
│   └── config.toml       # Streamlit configuration
├── README.md             # Documentation
└── LICENSE               # License file
```

## Dependencies

- streamlit: Web application framework
- torch: PyTorch for AI/ML capabilities
- midiutil: MIDI file generation
- pydub: Audio processing
- matplotlib: Plotting and visualizations
- numpy: Numerical computations
- Pillow: Image processing for album art
- plotly: Interactive visualizations
- scipy: Scientific computing

## Future Development

Planned features and improvements:

- [ ] Advanced AI music models (RNN, Transformer)
- [ ] Audio file export (WAV, MP3)
- [ ] User authentication and cloud storage
- [ ] Collaborative playlists
- [ ] Real-time audio playback
- [ ] More music genres and styles
- [ ] Enhanced lyric generation with rhyme schemes
- [ ] Advanced album art templates
- [ ] Mobile app version
- [ ] API for external integrations

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Music generation powered by AI algorithms
- Visualization libraries: Matplotlib and Plotly

## Support

For questions or issues:
- Open an issue on GitHub
- Use the in-app feedback system

---

**Built with ❤️ and AI** | Last updated: 2025-10-26
