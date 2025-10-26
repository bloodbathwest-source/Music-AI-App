# Music AI App - Quick Start Guide

## Installation

1. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Installation**
   ```bash
   python -c "from backend.database.models import init_db; init_db(); print('✅ Ready to run!')"
   ```

## Running the Application

### Option 1: Streamlit Web App (Recommended)

```bash
streamlit run app.py
```

The app will automatically:
- Initialize the database
- Open in your default browser at http://localhost:8501
- Be ready to generate music!

### Option 2: Backend API Only

```bash
python -m backend.api.main
```

Access the API at:
- API: http://localhost:8000
- Interactive Docs: http://localhost:8000/docs

## Using the Web App

### 1. Set Parameters
In the sidebar, configure:
- **Genre**: Pop, Jazz, Classical, Rock, Electronic, or Ambient
- **Mood**: Happy, Sad, Suspenseful, Energetic, Calm, or Dramatic
- **Key**: C, D, E, F, G, A, B (with sharps)
- **Mode**: Major, Minor, Dorian, etc.
- **Theme**: Optional descriptive theme

### 2. Advanced Settings (Optional)
- **Evolution Generations**: 10-100 (higher = better quality, slower)
- **Melody Length**: 16-64 notes

### 3. Generate Music
Click "🎼 Generate Music" and wait for the AI to evolve your composition.

### 4. Explore Results
- **Listen**: View waveform and download MIDI
- **Visualize**: Check note distribution and piano roll
- **Analyze**: See detailed note data

### 5. Provide Feedback
- Rate the music 1-5 stars
- Add optional comments
- Submit to help the AI learn and improve!

## API Usage

### Generate Music

```bash
curl -X POST "http://localhost:8000/api/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "genre": "pop",
    "mood": "happy",
    "key_root": "C",
    "mode": "major",
    "generations": 20
  }'
```

### Submit Feedback

```bash
curl -X POST "http://localhost:8000/api/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "music_id": 1,
    "rating": 5,
    "comment": "Amazing melody!"
  }'
```

### Get Statistics

```bash
curl "http://localhost:8000/api/stats"
```

## Understanding the Self-Learning Mechanism

1. **Generate Music**: AI creates music using evolutionary algorithms
2. **User Rates**: You provide feedback (1-5 stars)
3. **AI Learns**: System stores successful patterns
4. **Improve**: Future generations use learned patterns
5. **Iterate**: Each piece of feedback makes the AI better!

The AI tracks:
- Which genre/mood combinations get high ratings
- Successful parameter combinations
- User preferences over time

## File Locations

- **Generated MIDI Files**: `generated_music/`
- **Database**: `music_ai.db` (SQLite)
- **Logs**: Console output

## Troubleshooting

### Import Errors
```bash
pip install -r requirements.txt --upgrade
```

### Database Issues
Delete `music_ai.db` and restart the app to recreate it.

### Port Already in Use
- Streamlit: Use `streamlit run app.py --server.port 8502`
- API: Change port in `backend/api/main.py`

## Tips for Best Results

- **Start Simple**: Try 20 generations first
- **Experiment**: Different genre/mood combinations yield different styles
- **Provide Feedback**: More feedback = better AI
- **Download**: Save your favorite pieces as MIDI files
- **Be Patient**: Complex music takes time to evolve

## Support

For issues or questions:
1. Check the README.md for detailed documentation
2. Review the API documentation at http://localhost:8000/docs
3. Submit issues on GitHub

Enjoy creating music with AI! 🎵
