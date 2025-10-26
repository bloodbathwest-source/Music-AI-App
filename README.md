# Music-AI-App

A comprehensive AI-powered music generation web application built with Streamlit. Generate original melodies, create lyrics, and produce album art using advanced AI algorithms.

## Features

### 🎼 Music Generation
- **Genetic Algorithm-Based Melody Evolution**: Evolves melodies over multiple generations using fitness scoring
- **Multiple Musical Scales**: Support for major, minor, dorian, phrygian, lydian, mixolydian, and locrian modes
- **Customizable Parameters**:
  - Genre (pop, jazz, classical, rock)
  - Key root (C, D, E, F, G, A, B)
  - Mode/Scale
  - Emotion (happy, sad, suspenseful, energetic)
  - Tempo (60-180 BPM)
  - Evolution generations (10-100)
  - Melody length (16-64 notes)
- **MIDI File Export**: Download generated melodies as MIDI files
- **Melody Visualization**: Interactive charts showing note progression and dynamics
- **Statistics Display**: Total notes, duration, note range, and more

### 📝 Lyrics Generation
- **Theme-Based Creation**: Generate lyrics for different themes (love, sadness, adventure, hope)
- **Structured Song Format**: Includes verses, chorus, and bridge sections
- **Style-Specific Arrangements**: Adapts structure for pop, rock, jazz, and classical styles
- **Text File Export**: Download lyrics as formatted text files

### 🎨 Album Art Generation
- **Automatic Album Art**: AI-generated album artwork for both music and lyrics
- **Theme-Based Color Schemes**: Colors adapt to genre and theme
- **Professional Design**: Gradient backgrounds with geometric patterns

### ℹ️ Additional Features
- **Music Writers Table**: Display of respected composers and their notable works
- **User-Friendly Interface**: Clean, tabbed layout for easy navigation
- **Session State Management**: Preserve generated content during app usage
- **Responsive Design**: Works well on different screen sizes

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/bloodbathwest-source/Music-AI-App.git
   cd Music-AI-App
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

4. **Access the app**:
   Open your browser and navigate to `http://localhost:8501`

## Requirements

- Python 3.8 or higher
- Dependencies listed in `requirements.txt`:
  - streamlit>=1.28.0
  - torch>=2.6.0
  - matplotlib>=3.7.0
  - midiutil>=1.2.1
  - pydub>=0.25.1
  - numpy>=1.24.0
  - pillow>=10.3.0

## Project Structure

```
Music-AI-App/
├── app.py                      # Main Streamlit application
├── modules/                    # Backend modules
│   ├── __init__.py            # Package initialization
│   ├── music_generator.py     # Melody generation using genetic algorithms
│   ├── lyric_generator.py     # Lyric creation with theme-based templates
│   ├── midi_handler.py        # MIDI file creation and manipulation
│   └── visualization.py       # Plotting and album art generation
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── LICENSE                    # License file
```

## Usage

### Generating Music

1. Navigate to the **"Music Generator"** tab
2. Configure your parameters:
   - Select a genre (pop, jazz, classical, rock)
   - Choose your key and mode
   - Set the emotion and tempo
   - Adjust evolution generations for quality (more = better but slower)
   - Set melody length
3. Click **"Generate Music"**
4. View the generated melody visualization and album art
5. Download the MIDI file to use in your favorite music software

### Generating Lyrics

1. Navigate to the **"Lyrics Generator"** tab
2. Select your parameters:
   - Choose a theme (love, sadness, adventure, hope)
   - Select a style (pop, rock, jazz, classical)
   - Pick an emotion
3. Click **"Generate Lyrics"**
4. View the generated song with verses, chorus, and bridge
5. Download as a text file

### About Section

The **"About"** tab contains:
- Detailed information about the app's features
- Explanation of how the AI algorithms work
- Table of respected music writers and composers

## How It Works

### Music Generation Algorithm

The music generator uses a genetic algorithm approach:

1. **Initialization**: Creates a population of random melodies based on the selected scale
2. **Fitness Evaluation**: Each melody is scored based on:
   - Melodic contour (preference for stepwise motion)
   - Rhythmic variety
   - Dynamic range
   - Genre-specific characteristics
3. **Selection**: Top-performing melodies are selected
4. **Crossover**: Parent melodies are combined to create offspring
5. **Mutation**: Random changes introduce variation
6. **Iteration**: Process repeats for the specified number of generations
7. **Output**: Best melody is converted to MIDI format

### Lyric Generation

The lyric generator uses template-based generation:

1. **Theme Selection**: Chooses word banks based on the theme
2. **Structure Definition**: Determines song structure based on style
3. **Line Generation**: Creates lines using patterns and word combinations
4. **Section Assembly**: Combines lines into verses, choruses, and bridges
5. **Formatting**: Organizes into a complete song with proper labels

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Uses [PyTorch](https://pytorch.org/) for neural network capabilities
- MIDI creation powered by [MIDIUtil](https://github.com/MarkCWirt/MIDIUtil)
- Visualizations created with [Matplotlib](https://matplotlib.org/)

## Support

For issues, questions, or suggestions, please open an issue on the GitHub repository.

---

**Note**: This is an AI-powered tool designed for creative exploration and learning. The generated content is original and created by algorithms, not copied from existing works.
