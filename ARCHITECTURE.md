# Music AI App - Architecture Documentation

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Music AI App                             │
│                  (Streamlit Web Application)                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            ├─── app.py (Main Application)
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
   ┌────▼────┐         ┌────▼────┐        ┌────▼────┐
   │ Frontend │         │ Backend │        │  State  │
   │   (UI)   │         │ (Logic) │        │ Manager │
   └──────────┘         └──────────┘       └──────────┘
```

## Component Structure

### 1. Frontend Layer (UI Components)

```
Sidebar Navigation
├── Theme Toggle Button
├── Navigation Menu (7 pages)
│   ├── 🎹 AI Creator
│   ├── 📚 Library
│   ├── 🎧 Player
│   ├── �� Album Art
│   ├── ✍️ Lyrics
│   ├── 📊 Visualizations
│   └── 💬 Feedback
└── Quick Stats Display
    ├── Tracks Generated
    └── Feedback Received

Main Content Area
├── Page-Specific Content
├── Forms and Inputs
├── Visualizations
└── Download Buttons
```

### 2. Backend Layer (Logic & Processing)

```
Helper Functions
├── generate_music_data()
│   ├── Input: genre, key, mode, emotion, complexity
│   └── Output: melody list [(note, duration, velocity)]
│
├── create_midi_file()
│   ├── Input: melody, tempo
│   └── Output: MIDI BytesIO buffer
│
├── generate_lyrics()
│   ├── Input: theme, mood, num_lines
│   └── Output: formatted lyric string
│
├── create_album_art()
│   ├── Input: title, artist, color_scheme
│   └── Output: PIL Image object
│
├── create_waveform_viz()
│   ├── Input: melody
│   └── Output: Plotly Figure
│
└── create_piano_roll()
    ├── Input: melody
    └── Output: Matplotlib Figure
```

### 3. State Management

```
Session State Variables
├── library: List[Dict]
│   └── Track objects with metadata
├── current_track: Dict
│   └── Currently selected track
├── theme: str
│   └── 'dark' or 'light'
└── feedback: List[Dict]
    └── User feedback entries
```

## Data Flow

### Music Generation Flow

```
User Input
    ↓
[Select Parameters]
    ↓
generate_music_data()
    ↓
[Melody Array Generated]
    ↓
create_midi_file()
    ↓
[MIDI Buffer Created]
    ↓
┌──────────────────┐
│  Save to Library │
│  Display Preview │
│  Enable Download │
└──────────────────┘
```

### Visualization Flow

```
Track Selection
    ↓
[Get Melody Data]
    ↓
┌────────────┬────────────┬────────────┐
│  Waveform  │ Piano Roll │ Statistics │
├────────────┼────────────┼────────────┤
│  Plotly    │ Matplotlib │  Metrics   │
│  Chart     │   Chart    │   & Hists  │
└────────────┴────────────┴────────────┘
    ↓
[Display to User]
```

## Technology Stack

### Core Framework
```
Streamlit 1.28.0+
├── Web server
├── Auto-reload
├── Session management
└── Component rendering
```

### Data Processing
```
NumPy 1.24.3+
├── Array operations
├── Statistical calculations
└── Numerical processing
```

### Visualizations
```
Plotly 5.17.0+          Matplotlib 3.7.2+
├── Interactive charts  ├── Static plots
├── Zoom/Pan           ├── Piano rolls
└── Hover tooltips     └── Histograms
```

### File Generation
```
MIDIUtil 1.2.1+         Pillow 10.0.0+
├── MIDI creation      ├── Image creation
├── Note sequences     ├── Drawing tools
└── File export        └── PNG export
```

## Deployment Architecture

### Streamlit Cloud Deployment

```
GitHub Repository
    ↓
Streamlit Cloud
    ├── Auto-build
    ├── Environment setup
    ├── Dependency installation
    └── App deployment
    ↓
Live Application
    ├── HTTPS enabled
    ├── Auto-scaling
    └── CDN distribution
```

### Heroku Deployment

```
GitHub Repository
    ↓
Heroku Platform
    ├── Buildpack detection
    ├── Procfile execution
    ├── Environment variables
    └── Dyno allocation
    ↓
Live Application
    ├── Custom domain
    ├── SSL certificate
    └── Log aggregation
```

## Security Architecture

```
Application Security
├── Input Validation
│   ├── Parameter bounds checking
│   ├── Type validation
│   └── Length restrictions
├── Session Isolation
│   ├── Browser-based storage
│   ├── No cross-session data
│   └── Automatic cleanup
├── XSS Protection
│   ├── Streamlit auto-escaping
│   └── Safe HTML rendering
└── No Sensitive Data
    ├── No passwords
    ├── No API keys
    └── No personal info
```

## Performance Architecture

### Optimization Strategies

```
Caching (Future)
├── @st.cache_data for expensive operations
├── Melody generation results
└── Visualization objects

Session Management
├── Limit library to 50 tracks (recommended)
├── Periodic cleanup of old data
└── Browser storage limits

Async Operations
├── Non-blocking UI updates
├── Progress indicators
└── Spinner animations
```

### Resource Usage

```
Memory Usage
├── App Base: ~50MB
├── Per Track: ~10KB
├── Visualizations: ~5MB
└── Total (50 tracks): ~100MB

Network Usage
├── Initial Load: ~2MB
├── Static Assets: ~500KB
├── Per Generation: <100KB
└── Minimal data transfer
```

## Module Dependencies

```
app.py
├── io (standard library)
├── random (standard library)
├── datetime (standard library)
├── matplotlib.pyplot
├── numpy
├── plotly.graph_objects
├── streamlit
├── midiutil.MIDIFile
└── PIL.Image, PIL.ImageDraw
```

## File Structure

```
Music-AI-App/
├── app.py                      # Main application (575 lines)
├── requirements.txt            # Dependencies (6 packages)
├── Procfile                    # Heroku config
├── .streamlit/
│   └── config.toml            # Streamlit settings
├── README.md                   # Project overview
├── USER_GUIDE.md              # End-user documentation
├── DEPLOYMENT.md              # Deployment guides
├── FEATURES.md                # Feature documentation
├── IMPLEMENTATION_SUMMARY.md  # Project summary
└── ARCHITECTURE.md            # This file

Generated at Runtime (not committed):
├── __pycache__/               # Python cache
└── *.mid                      # Generated MIDI files
```

## API Surface (Internal)

### Music Generation API

```python
generate_music_data(
    genre: str,        # 'pop', 'jazz', 'classical', 'rock', 'electronic'
    key_root: str,     # 'C', 'D', 'E', 'F', 'G', 'A', 'B'
    mode: str,         # 'major', 'minor', 'dorian'
    emotion: str,      # 'happy', 'sad', 'suspenseful', 'energetic'
    complexity: str    # 'Simple', 'Medium', 'Complex'
) -> List[Tuple[int, float, int]]  # [(note, duration, velocity), ...]
```

### MIDI Creation API

```python
create_midi_file(
    melody: List[Tuple[int, float, int]],
    tempo: int = 120
) -> io.BytesIO  # MIDI file buffer
```

### Lyrics Generation API

```python
generate_lyrics(
    theme: str,      # 'love', 'nature', 'life', 'adventure'
    mood: str,       # 'happy', 'sad', 'energetic', 'contemplative'
    num_lines: int   # 4-16
) -> str  # Formatted lyrics text
```

### Album Art API

```python
create_album_art(
    title: str,
    artist: str,
    color_scheme: List[Tuple[int, int, int]]
) -> PIL.Image.Image  # 400x400 RGB image
```

### Visualization APIs

```python
create_waveform_viz(
    melody: List[Tuple[int, float, int]]
) -> plotly.graph_objects.Figure

create_piano_roll(
    melody: List[Tuple[int, float, int]]
) -> matplotlib.figure.Figure
```

## Event Flow

### User Interaction Flow

```
1. User Action (button click, input change)
    ↓
2. Streamlit Event Handler
    ↓
3. State Update (if needed)
    ↓
4. Function Execution
    ↓
5. Result Processing
    ↓
6. UI Update/Re-render
    ↓
7. Display to User
```

### Session Lifecycle

```
App Start
    ↓
Initialize Session State
    ├── library = []
    ├── current_track = None
    ├── theme = 'dark'
    └── feedback = []
    ↓
User Interactions
    ├── Generate music
    ├── Navigate pages
    ├── Submit feedback
    └── Toggle theme
    ↓
Session End (browser close)
    └── Data cleared
```

## Scalability Considerations

### Current Limits

```
Browser Session Storage
├── Recommended: 50 tracks
├── Maximum: 100+ tracks
└── Limit: Browser memory

Performance
├── Generation: O(n) where n = notes
├── Search: O(m) where m = tracks
└── Visualization: O(n) for data points
```

### Future Scaling Options

```
Backend Storage
├── Database integration
├── Cloud file storage
└── User accounts

Caching Layer
├── Redis for session data
├── CDN for static assets
└── Result caching

Load Balancing
├── Multiple app instances
├── Geographic distribution
└── Auto-scaling
```

## Error Handling

### Current Implementation

```
Input Validation
├── Bounded sliders (no invalid tempo)
├── Dropdown selections (no invalid genres)
└── Required fields (enforced by UI)

Graceful Degradation
├── Empty library handling
├── No track selected handling
└── Missing data fallbacks

User Feedback
├── Success messages (green)
├── Info messages (blue)
└── Spinner animations (loading)
```

## Browser Compatibility

```
Supported Browsers
├── Chrome 90+ ✅
├── Firefox 88+ ✅
├── Safari 14+ ✅
├── Edge 90+ ✅
└── Mobile browsers ✅

Required Features
├── JavaScript enabled
├── Modern CSS support
└── WebSocket support (for Streamlit)
```

## Development Workflow

```
Code Changes
    ↓
Local Testing
    ↓
Git Commit
    ↓
Push to GitHub
    ↓
Automatic Deployment (if enabled)
    ↓
Production Update
```

---

**Architecture Version**: 1.0.0
**Last Updated**: 2025-10-26
**Status**: Production Ready
