# Music AI App

A cutting-edge **Music AI Application** that enables users to create, edit, and share AI-generated music with advanced features.

## Overview

This application provides a complete music AI platform with:
- **AI Music Generation**: Create music using deep learning models with customizable parameters
- **User Authentication**: Secure login and user management
- **Music Editing**: Multi-track editing capabilities
- **Export**: Download tracks as MIDI or WAV (platform sharing to Spotify, SoundCloud, and YouTube is on the roadmap — not built yet)
- **Cloud Storage**: Save and manage your music library

The Streamlit app (`app.py`) plays every generated track right in the
browser, keeps your tracks in a personal collection, and lets you download
a backup of your tracks and reward progress (Settings → Data Management)
to restore later.

## Quick Start

### Option 1: Run the Streamlit App

For a quick demo of the music generation capabilities:

```bash
pip install -r requirements.txt
streamlit run app.py
```

This will launch a web interface for music generation with genre, key,
mode, and emotion controls.

#### Using the Streamlit App

1. **Create a track** on the "🎼 Create Music" page: pick a genre
   (pop, jazz, classical, rock), key, mode, emotion, generations
   (10–100; higher = more evolved), and tempo, then click
   **"🎹 Generate Music"**.
2. **Listen immediately** — the generated track plays in the browser, and
   can be downloaded as MIDI or WAV. Every track you generate is kept in
   **My Collection** at the bottom of the page, where it can be replayed
   or removed.
3. **Earn rewards** — creating tracks unlocks achievements, silly titles,
   and collectibles; check them on the "🎮 Profile & Rewards" page.
   (Rewards can be turned off in Settings.)
4. **Back up your collection** — session data disappears when the tab
   closes, so download a backup from **⚙️ Settings → Data Management**
   and restore it there on your next visit.

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
- Download generated music files as MIDI (Streamlit app and backend API)
  or WAV (Streamlit app, rendered in-process from the generated notes)
- MP3/FLAC export and platform sharing to Spotify, SoundCloud, and
  YouTube are on the roadmap — not built yet

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

## Configuration

The backend is configured through environment variables (see
`.env.example` for the full list). The essentials:

```env
# Database (PostgreSQL via SQLAlchemy; DATABASE_URL is derived from these)
POSTGRES_SERVER=localhost
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_DB=musicai

# MongoDB (music metadata)
MONGODB_URL=mongodb://localhost:27017
MONGODB_DB=musicai_metadata

# Security
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

## Testing

Run the test suite:

```bash
# Everything (backend API + Streamlit collection helpers)
pytest

# Backend tests only
pytest tests/backend/

# With coverage
pytest tests/backend/ --cov=backend
```

## Documentation

This README is the project's front door. Older, overlapping documents
(the former full README, quick-start guide, implementation summary, and
reward-system doc) were consolidated into it on 2026-11-26; their
original texts are preserved verbatim under
[docs/archive/](docs/archive/). The original Streamlit prototype,
formerly the root file `code version one`, is preserved at
[archive/code version one](archive/code%20version%20one).

## Development Roadmap

- **Phase 1 (Current)**: Core API and basic AI generation
- **Phase 2**: Advanced AI models, multi-track editor, real-time collaboration
- **Phase 3**: Marketplace, mobile apps, custom AI training
- **Phase 4**: Production deployment and scaling

## License

See LICENSE file for details
