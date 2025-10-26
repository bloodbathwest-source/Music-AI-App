# 🎵 Music AI App

A comprehensive music creation and management application powered by AI. Generate original music, lyrics, and album art with customizable parameters, manage your music library, and visualize your creations.

## ✨ Features

### 🤖 AI-Powered Creation
- **Music Generation**: Create original MIDI compositions with customizable parameters
  - Genre selection (Pop, Rock, Jazz, Classical, Electronic)
  - Key and mode selection
  - Tempo control (60-180 BPM)
  - Emotion-based generation (Happy, Sad, Energetic, Calm, Suspenseful)
  
- **Lyric Generation**: AI-generated lyrics with verse-chorus-bridge structure
  - Genre-specific templates
  - Emotion-based word selection
  - Editable and downloadable output

- **Album Art Creation**: Auto-generated geometric album covers
  - Color schemes based on emotion and genre
  - Custom title overlay
  - PNG export

### 📚 Music Library
- Comprehensive song database with SQLite backend
- Grid and list view modes
- Sorting by date, title, and genre
- Playlist creation and management
- CRUD operations for all music items

### 🔍 Search & Filter
- Search by title, artist, or genre
- Filter by genre and key
- Multiple sorting options
- Real-time results

### 📊 Interactive Visualizations
- Waveform visualization with amplitude analysis
- Music notation display
- Genre profile radar charts
- Musical analysis with audio features

### 🎨 User Experience
- Dark/Light theme toggle
- Responsive design
- Intuitive navigation
- Real-time music preview
- Download capabilities for MIDI and lyrics

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/bloodbathwest-source/Music-AI-App.git
cd Music-AI-App
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

4. Open your browser and navigate to `http://localhost:8501`

### Using with Dev Containers

This repository includes a Dev Container configuration for easy setup:

1. Open the repository in VS Code
2. Install the "Remote - Containers" extension
3. Click "Reopen in Container" when prompted
4. The app will start automatically

## 📖 Usage Guide

### Generating Your First Song

1. **Navigate to AI Creator** (🤖 icon in sidebar)
2. **Choose Full Song tab** for complete generation
3. **Set parameters**:
   - Enter a title for your song
   - Select genre and emotion
   - Choose musical key and mode
   - Set tempo (BPM)
4. **Click "Generate Full Song"**
5. **Preview and download** your creation

### Managing Your Library

1. **Navigate to Library** (🎵 icon in sidebar)
2. **Browse songs** in grid or list view
3. **Play songs** by clicking the play button
4. **Create playlists** using the expandable section
5. **Delete songs** if needed

### Searching for Music

1. **Navigate to Search** (🔍 icon in sidebar)
2. **Enter search terms** (title, artist, or genre)
3. **Apply filters** for genre and key
4. **Sort results** by different criteria
5. **Play songs** directly from search results

### Viewing Visualizations

1. **Select a song** to play
2. **Navigate to Visualizations** (📊 icon in sidebar)
3. **Explore different views**:
   - Waveform tab for amplitude visualization
   - Notation tab for musical parameters
   - Analysis tab for genre profiling

## 🏗️ Project Structure

```
Music-AI-App/
├── app.py                      # Main application entry point
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── backend/                    # AI generation services
│   ├── __init__.py
│   ├── music_generator.py      # MIDI music generation
│   ├── lyric_generator.py      # Lyric creation
│   └── album_art_generator.py  # Album art creation
│
├── components/                 # UI components
│   ├── __init__.py
│   ├── player.py              # Music player interface
│   ├── library.py             # Library management
│   ├── search.py              # Search functionality
│   ├── ai_creator.py          # AI creation interface
│   └── visualizations.py      # Visualization components
│
├── database/                   # Database layer
│   ├── __init__.py
│   ├── models.py              # SQLAlchemy models
│   └── crud.py                # Database operations
│
└── static/                    # Static assets
    ├── audio/                 # Generated MIDI files
    └── images/                # Generated album art
```

## 🛠️ Technical Stack

- **Framework**: Streamlit (Python web framework)
- **Database**: SQLite with SQLAlchemy ORM
- **Music Generation**: MIDIUtil, Music21
- **Visualizations**: Matplotlib, NumPy
- **Image Processing**: Pillow (PIL)
- **Audio Processing**: PyDub (optional for advanced features)

## 🎯 API Documentation

### Database Models

#### Song
```python
{
    'id': int,
    'title': str,
    'artist': str,
    'genre': str,
    'duration': float,
    'file_path': str,
    'lyrics': str,
    'album_art_path': str,
    'key': str,
    'tempo': int,
    'is_ai_generated': bool,
    'created_at': datetime
}
```

#### Playlist
```python
{
    'id': int,
    'name': str,
    'description': str,
    'song_ids': str,  # Comma-separated IDs
    'created_at': datetime
}
```

### Backend Services

#### MusicGenerator
```python
generator = MusicGenerator(
    genre='pop',
    key='C',
    mode='major',
    emotion='happy',
    tempo=120
)
midi_path = generator.generate_midi(output_path, title)
```

#### LyricGenerator
```python
generator = LyricGenerator(genre='pop', emotion='happy')
lyrics = generator.generate_full_lyrics()
```

#### AlbumArtGenerator
```python
generator = AlbumArtGenerator(emotion='happy', genre='pop')
image_path = generator.generate(title, output_path)
```

## 🔄 State Management

The app uses Streamlit's session state for managing:
- Current playing song ID
- Theme preference (dark/light)
- Player actions (play, pause, next, previous)
- Shuffle and repeat modes
- Search filters and results

## 🎨 Customization

### Adding New Genres

Edit `backend/music_generator.py` and `backend/lyric_generator.py` to add genre-specific patterns:

```python
# In music_generator.py
EMOTION_PATTERNS = {
    'your_genre': {
        'tempo_range': (100, 130),
        'velocity_range': (70, 100),
        'note_density': 0.6
    }
}

# In lyric_generator.py
VERSE_STRUCTURES = {
    'your_genre': [
        "Template with {placeholders}",
        # Add more templates
    ]
}
```

### Customizing Themes

Edit the `apply_custom_css()` function in `app.py`:

```python
if theme == 'your_theme':
    primary_color = "#YOUR_COLOR"
    background_color = "#YOUR_BG"
    secondary_bg = "#YOUR_SECONDARY"
```

## 🐛 Troubleshooting

### Common Issues

**Database errors on startup:**
- Delete `database/music_ai.db` and restart the app
- The database will be recreated automatically

**MIDI files not playing:**
- Ensure your browser supports MIDI playback
- Use Chrome or Firefox for best compatibility
- Download the MIDI file and play in a local player

**Missing dependencies:**
```bash
pip install -r requirements.txt --upgrade
```

**Port already in use:**
```bash
streamlit run app.py --server.port 8502
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Music generation inspired by evolutionary algorithms
- UI design following modern web app conventions
- Built with Streamlit framework

## 📧 Support

For support, please open an issue in the GitHub repository or contact the maintainers.

---

**Version:** 1.0.0  
**Last Updated:** 2025-10-26  
**Status:** Production Ready for Trial Use
