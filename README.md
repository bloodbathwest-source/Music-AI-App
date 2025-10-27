# Music AI App

A cutting-edge **Music AI Application** that enables users to create, edit, and share AI-generated music with advanced features.

## Overview

This application provides a complete music AI platform with:
- **AI Music Generation**: Create music using deep learning models with customizable parameters
- **User Authentication**: Secure login and user management
- **Music Editing**: Multi-track editing capabilities
- **Export & Sharing**: Export to various formats and share to platforms like Spotify, SoundCloud, and YouTube
- **Cloud Storage**: Save and manage your music library

## Quick Start

### Option 1: Run the Streamlit Prototype

For a quick demo of the music generation capabilities:

```bash
pip install -r requirements.txt
streamlit run app.py
```

This will launch a simple web interface for music generation with genre, key, mode, and emotion controls.

### Option 2: Full Application (Backend + Frontend)

#### Backend Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your configuration

# Start the FastAPI server
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000` with interactive docs at `http://localhost:8000/docs`

#### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at `http://localhost:3000`

## Features

### 1. Music Creation (AI Generation)
- Style selection for genres (pop, jazz, classical, rock, electronic, ambient)
- Mood selection (happy, sad, energetic, calm, suspenseful)
- Custom inputs like tempo (60-200 BPM), key (C, D, E, F, G, A, B), and duration (30-180 seconds)
- AI-generated tracks using LSTM neural networks
- MIDI export for DAW integration

### 2. User Accounts and Authentication
- Email-based registration and login
- JWT token authentication
- User profile management
- Secure password hashing

### 3. Music Management
- List and organize your generated tracks
- Delete unwanted tracks
- Share tracks publicly or privately
- Track metadata management

### 4. Export and Sharing
- Export support for MP3, WAV, FLAC, and MIDI formats
- Platform integration support for Spotify, SoundCloud, and YouTube
- Download generated music files

## Architecture

- **Backend**: FastAPI (Python) with async support
- **Frontend**: Next.js (React) with Material-UI
- **AI/ML**: PyTorch, TensorFlow, Magenta for music generation
- **Databases**: PostgreSQL for user data, MongoDB for music metadata
- **Authentication**: JWT with OAuth support
- **Storage**: AWS S3 / Google Cloud Storage integration

## API Endpoints

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/token` - Login
- `GET /api/auth/me` - Get current user

### Music Generation
- `POST /api/music/generate` - Generate AI music
- `GET /api/music/generate/{task_id}` - Check generation status
- `GET /api/music/tracks` - List tracks
- `GET /api/music/tracks/{track_id}` - Get track details
- `DELETE /api/music/tracks/{track_id}` - Delete track

### Export
- `POST /api/export/export` - Export track
- `POST /api/export/upload` - Upload to platform

## Testing

Run the test suite:

```bash
# Backend tests
pytest tests/backend/

# With coverage
pytest tests/backend/ --cov=backend
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

When creating issues, please use our issue templates:
- **Bug Report**: For reporting bugs
- **Feature Request**: For suggesting new features  
- **Task for Copilot**: For creating well-scoped tasks optimized for GitHub Copilot coding agent

## Documentation

For detailed documentation, see [README_FULL.md](README_FULL.md)

For information about using GitHub Copilot coding agent with this repository, see [COPILOT_GUIDE.md](COPILOT_GUIDE.md)

## Development Roadmap

- **Phase 1 (Current)**: Core API and basic AI generation
- **Phase 2**: Advanced AI models, multi-track editor, real-time collaboration
- **Phase 3**: Marketplace, mobile apps, custom AI training
- **Phase 4**: Production deployment and scaling

## License

See LICENSE file for details
