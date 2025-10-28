# Music-AI-App

A self-contained Streamlit web app for creating AI-generated music with an engaging reward system. This application allows users to generate music by selecting parameters (genre, key, mode, emotion, generations) and provides gamification features to enhance the user experience.

## Features

### Music Generation
- **Genre Selection**: Choose from pop, jazz, classical, or rock
- **Musical Parameters**: Select key, mode, and emotional tone
- **Evolutionary Algorithm**: Generate music using genetic algorithms
- **MIDI Export**: Download generated music as MIDI files
- **Visualization**: View melody visualizations

### Reward System
- **Achievements**: Unlock achievements for various activities
- **Silly Titles**: Earn and display humorous titles
- **Virtual Collectibles**: Collect badges and stickers with different rarities
- **Fun Rewards**: Receive music jokes, trivia, and tips
- **Level System**: Progress through levels based on activity
- **Customizable**: Enable/disable rewards for a minimalistic experience

For detailed information about the reward system, see [REWARD_SYSTEM.md](REWARD_SYSTEM.md).

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

## Requirements

- Python 3.8+
- streamlit
- torch
- matplotlib
- midiutil
- pydub
