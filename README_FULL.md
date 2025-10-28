# Music AI App

A cutting-edge Music AI Application that enables users to create, edit, and share AI-generated music.

## Features

### 1. Music Creation (AI Generation)
- Style selection for genres, moods, and instruments
- Custom inputs like tempo, key, and melody inspiration
- AI-generated tracks using deep learning models
- Track length options (30 seconds to 3 minutes)

### 2. Editing and Customization
- Multi-track editor for volume, pan, and instrument adjustments
- Effect plugins such as reverb, delay, and equalization
- MIDI export for DAW integration

### 3. Music Uploading and Sharing
- Integrated uploads to platforms like Spotify, SoundCloud, and YouTube
- Metadata auto-filling (artist name, genre, album)
- Export support for MP3, WAV, FLAC, and MIDI formats
- Private or collaborative sharing options

### 4. User Accounts and Cloud Sync
- Authentication via email and OAuth providers
- Cloud storage for drafts and completed tracks
- Profile dashboards for music management

### 5. Community and Collaboration
- Music marketplace for buying and selling tracks
- Collaboration tools for shared editing

## Architecture

### Backend
- **Framework**: FastAPI (Python)
- **AI/ML**: PyTorch, TensorFlow, Magenta
- **Databases**: 
  - PostgreSQL for user data
  - MongoDB for music metadata
- **Storage**: AWS S3 / Google Cloud Storage
- **Authentication**: JWT-based with OAuth support

### Frontend
- **Framework**: Next.js (React)
- **UI Library**: Material-UI
- **Audio**: Tone.js, WaveSurfer.js
- **State Management**: React Hooks

### AI Services
- Music generation using LSTM networks
- Genre-based style transfer
- Evolutionary algorithms for melody generation

## Installation

### Prerequisites
- Python 3.8+
- Node.js 16+
- PostgreSQL 13+
- MongoDB 5+

### Backend Setup

```bash
# Install Python dependencies
pip install -r requirements_clean.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Run database migrations
alembic upgrade head

# Start the API server
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Set up environment variables
cp .env.local.example .env.local
# Edit .env.local with your API URL

# Start development server
npm run dev
```

### Streamlit App (Legacy)

The original Streamlit prototype is available in `app.py`:

```bash
streamlit run app.py
```

## API Documentation

Once the backend is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Key Endpoints

#### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/token` - Login and get access token
- `GET /api/auth/me` - Get current user info

#### Music Generation
- `POST /api/music/generate` - Generate AI music
- `GET /api/music/generate/{task_id}` - Check generation status
- `GET /api/music/tracks` - List user tracks
- `GET /api/music/tracks/{track_id}` - Get track details
- `DELETE /api/music/tracks/{track_id}` - Delete track
- `POST /api/music/tracks/{track_id}/share` - Share/unshare track

#### Export & Sharing
- `POST /api/export/export` - Export track in format
- `GET /api/export/download/{filename}` - Download exported track
- `POST /api/export/upload` - Upload to platform (Spotify, SoundCloud, YouTube)
- `GET /api/export/upload/{upload_id}/status` - Check upload status

#### User Management
- `GET /api/users/profile` - Get user profile
- `PUT /api/users/profile` - Update user profile
- `GET /api/users/tracks` - Get user's tracks

## Project Structure

```
Music-AI-App/
├── backend/
│   └── app/
│       ├── api/          # API endpoints
│       ├── core/         # Core configuration and security
│       ├── models/       # Database models
│       ├── services/     # Business logic and AI services
│       └── main.py       # FastAPI application
├── frontend/
│   ├── src/
│   │   ├── components/   # React components
│   │   ├── pages/        # Next.js pages
│   │   ├── services/     # API service layer
│   │   └── styles/       # CSS styles
│   ├── package.json
│   └── next.config.js
├── tests/
│   ├── backend/          # Backend tests
│   └── frontend/         # Frontend tests
├── app.py                # Legacy Streamlit app
├── requirements_clean.txt # Python dependencies (clean version)
├── requirements.txt      # Python dependencies (legacy reference)
└── README.md
```

## Development

### Running Tests

Backend:
```bash
pytest tests/backend/
```

Frontend:
```bash
cd frontend
npm test
```

### Code Quality

Python linting:
```bash
pylint backend/
```

## Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# Database
POSTGRES_SERVER=localhost
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_DB=musicai

MONGODB_URL=mongodb://localhost:27017
MONGODB_DB=musicai_metadata

# Security
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# AWS (optional)
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_BUCKET_NAME=musicai-storage
AWS_REGION=us-east-1

# API URLs
API_V1_STR=/api/v1
```

### Frontend Environment Variables

Create `.env.local` in the `frontend/` directory:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000/api
```

## Deployment

### Docker Support (Coming Soon)
Docker and docker-compose configurations will be added for easy deployment.

### Cloud Deployment
- Backend: AWS Lambda, Google Cloud Run, or traditional VPS
- Frontend: Vercel, Netlify, or AWS S3 + CloudFront
- Database: AWS RDS (PostgreSQL), MongoDB Atlas

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

See LICENSE file for details.

## Roadmap

### Phase 1 (Current)
- [x] Core API structure
- [x] Basic AI music generation
- [x] User authentication
- [x] Frontend scaffolding

### Phase 2
- [ ] Advanced AI models integration
- [ ] Multi-track editor
- [ ] Real-time collaboration
- [ ] Platform integrations (Spotify, SoundCloud)

### Phase 3
- [ ] Music marketplace
- [ ] Advanced effect plugins
- [ ] Mobile apps (React Native)
- [ ] Custom AI model training

### Phase 4
- [ ] Scalability optimizations
- [ ] Advanced analytics
- [ ] Community features
- [ ] Production deployment

## Support

For issues and questions, please use the GitHub issue tracker.

## Acknowledgments

- Magenta (Google) for music generation models
- OpenAI for inspiration
- The open-source music AI community
