# Changelog

All notable changes to the Music AI App will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2025-10-27

### Security
- **Fixed critical security vulnerabilities**
  - Updated python-multipart from 0.0.12 to 0.0.18 (CVE: DoS via malformed multipart/form-data boundary)
  - Updated python-jose from 3.3.0 to 3.4.0 (CVE: Algorithm confusion with OpenSSH ECDSA keys)
  - Updated torch from 2.1.1 to 2.6.0 (CVE: Remote code execution via torch.load)

### Changed
- **Updated dependencies for Python 3.12 compatibility**
  - Updated torch from 2.1.1 to 2.6.0 (Python 3.12 support + security fix)
  - Updated numpy from 1.24.3 to 1.26.4 (required for Python 3.12)
  - Updated scipy from 1.11.4 to 1.13.1 (better numpy 1.26 compatibility)
  - Updated fastapi from 0.104.1 to 0.115.0 (security and bug fixes)
  - Updated uvicorn from 0.24.0 to 0.32.0
  - Updated streamlit from 1.28.2 to 1.39.0
  - Updated pydantic from 2.5.0 to 2.9.2
  - Updated pytest from 7.4.3 to 8.3.3
  - Updated other dependencies to latest stable versions
- **Removed incompatible dependencies**
  - Removed tensorflow (2.15.0) - Python 3.12 support incomplete
  - Removed magenta (2.1.4) - depends on tensorflow
- **Updated Dockerfile.backend**
  - Changed base image from Python 3.11 to Python 3.12
  - Added system dependencies: libsndfile1, ffmpeg, portaudio19-dev
  - Added make to build tools

### Added
- **INSTALL.md** - Comprehensive installation guide
  - System requirements and dependencies
  - Platform-specific installation instructions (Ubuntu, macOS, Windows)
  - Python 3.12 compatibility notes
  - Troubleshooting section
  - Docker installation instructions
- **Enhanced requirements.txt** - Added detailed comments explaining:
  - Version choices and compatibility notes
  - System dependencies required by packages
  - Reasons for major version changes
  - Alternative installation options for PyTorch

### Fixed
- Installation errors on Python 3.12 due to outdated package versions
- Missing system dependency documentation
- Incomplete setup instructions in README files

### Documentation
- Updated README.md to reference INSTALL.md
- Updated README_FULL.md with Python 3.12 requirements
- Added virtual environment setup instructions
- Documented system-level dependencies (libsndfile1, ffmpeg, etc.)
- Added troubleshooting steps for common installation issues

### Notes
- Python 3.12 is now the recommended version (Python 3.11 still supported)
- TensorFlow and Magenta support will be reconsidered when Python 3.12 compatibility improves
- The application remains fully functional without TensorFlow/Magenta (uses PyTorch for ML)

## [1.0.0] - 2025-10-26

### Added
- Initial release of Music AI App
- FastAPI backend with REST API
- Authentication system (JWT-based)
- Music generation service with AI
  - Multiple genre support (pop, jazz, classical, rock, electronic, ambient)
  - Mood-based generation (happy, sad, energetic, calm, suspenseful)
  - Customizable tempo (60-200 BPM)
  - Key selection (all major and minor keys)
  - Duration control (30-180 seconds)
- MIDI export functionality
- React/Next.js frontend
  - Music generator UI with Material-UI
  - API service layer
- User management
  - Registration and login
  - Profile management
  - Track library
- Export and sharing capabilities
  - Multiple format support (MP3, WAV, FLAC, MIDI)
  - Platform integration placeholders (Spotify, SoundCloud, YouTube)
- Comprehensive test suite (18 tests)
- Documentation
  - README with quick start guide
  - Detailed README_FULL.md
  - Contributing guidelines
  - Environment configuration examples
- Docker and Docker Compose support
- Database models
  - PostgreSQL for user data
  - MongoDB for music metadata
- Security features
  - Password hashing (bcrypt)
  - JWT token authentication
  - CORS configuration

### Technical Details
- Python 3.8+ backend
- Node.js 16+ frontend
- FastAPI for REST API
- SQLAlchemy for PostgreSQL ORM
- PyMongo for MongoDB
- Material-UI for frontend components
- Pytest for testing

## [Unreleased]

### Planned Features
- Advanced AI models integration (Magenta, Jukebox)
- Multi-track editor
- Real-time collaboration
- Music marketplace
- Mobile applications (React Native)
- Custom AI model training
- Advanced audio effects
- Social features and community
- Analytics dashboard
- Payment integration (Stripe, PayPal)
