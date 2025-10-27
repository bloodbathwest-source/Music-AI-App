# Installation Guide

This guide provides detailed instructions for installing and setting up the Music AI App.

## System Requirements

### Python Version
- **Python 3.11 or 3.12** (recommended: Python 3.12)
- Python 3.10 may work but is not officially supported
- Python 3.8-3.9 are not supported due to dependency requirements

### System Dependencies

Before installing Python dependencies, ensure the following system-level packages are installed:

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y \
    python3-dev \
    python3-pip \
    gcc \
    g++ \
    make \
    libpq-dev \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    portaudio19-dev
```

#### macOS
```bash
brew install python@3.12
brew install postgresql
brew install libsndfile
brew install ffmpeg
brew install portaudio
```

#### Windows
- Install Python 3.12 from [python.org](https://www.python.org/downloads/)
- Install Visual Studio Build Tools (for compiling some packages)
- Install PostgreSQL from [postgresql.org](https://www.postgresql.org/download/windows/)
- Install ffmpeg from [ffmpeg.org](https://ffmpeg.org/download.html)

### Database Requirements

#### PostgreSQL
- PostgreSQL 13 or higher
- Used for user data and application state

#### MongoDB
- MongoDB 5 or higher
- Used for music metadata storage

## Python Dependencies Installation

### 1. Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python3.12 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 2. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Installation Notes

#### PyTorch Installation
If you encounter issues with PyTorch installation on certain platforms:

```bash
# For CPU-only installation (smaller, faster)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# For CUDA 11.8 (for NVIDIA GPU support)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1 (for newer NVIDIA GPU support)
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

#### Audio Processing Dependencies
The `soundfile` and `librosa` packages require `libsndfile` to be installed at the system level (see System Dependencies above).

If you encounter issues with audio processing:
```bash
# Verify libsndfile is installed
python -c "import soundfile; print(soundfile.__version__)"
```

#### TensorFlow and Magenta
**Note**: TensorFlow and Magenta have been removed from the requirements due to:
- Limited Python 3.12 support at the time of writing
- Not actively used in the current codebase
- Large installation size

If you need TensorFlow or Magenta for custom extensions:
- Use Python 3.11 instead of 3.12
- Or wait for official Python 3.12 support from these projects

## Dependency Notes

### Major Version Updates (Python 3.12 Compatibility)

The following packages have been updated for Python 3.12 compatibility:

| Package | Old Version | New Version | Reason |
|---------|------------|-------------|--------|
| torch | 2.1.1 | 2.5.1 | Python 3.12 support starts at 2.2.0 |
| numpy | 1.24.3 | 1.26.4 | Python 3.12 requires 1.26+ |
| scipy | 1.11.4 | 1.13.1 | Better compatibility with numpy 1.26+ |
| tensorflow | 2.15.0 | Removed | Python 3.12 support incomplete |
| magenta | 2.1.4 | Removed | Depends on tensorflow |
| fastapi | 0.104.1 | 0.115.0 | Security and bug fixes |
| streamlit | 1.28.2 | 1.39.0 | Latest stable version |
| pydantic | 2.5.0 | 2.9.2 | Better FastAPI integration |

## Verification

After installation, verify that all dependencies are installed correctly:

```bash
# Check Python version
python --version

# Check that critical packages are importable
python -c "import fastapi; print('FastAPI:', fastapi.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "import librosa; print('Librosa:', librosa.__version__)"
python -c "import streamlit; print('Streamlit:', streamlit.__version__)"
```

## Troubleshooting

### Common Issues

#### 1. `pip install` fails with network errors
```bash
# Increase timeout
pip install --timeout=1000 -r requirements.txt

# Or install packages individually
pip install fastapi uvicorn
pip install torch
pip install numpy scipy
# ... etc
```

#### 2. `soundfile` import fails
```bash
# Make sure libsndfile is installed
# Ubuntu/Debian:
sudo apt-get install libsndfile1

# macOS:
brew install libsndfile
```

#### 3. PostgreSQL connection issues
- Verify PostgreSQL is running: `sudo systemctl status postgresql`
- Check your `.env` file has correct database credentials
- Ensure the database exists: `createdb musicai`

#### 4. Compilation errors during installation
```bash
# Make sure build tools are installed
# Ubuntu/Debian:
sudo apt-get install build-essential python3-dev

# macOS:
xcode-select --install
```

### Memory Issues During Installation

Some packages (especially PyTorch) require significant memory during installation:

```bash
# Install one package at a time if memory is limited
pip install numpy
pip install torch
pip install librosa
# ... continue with other packages
```

## Docker Installation (Alternative)

For a containerized installation that includes all system dependencies:

```bash
# Build the backend container
docker build -f Dockerfile.backend -t music-ai-backend .

# Run with docker-compose
docker-compose up
```

Note: You'll need to update the Dockerfile.backend to include additional system dependencies listed above.

## Next Steps

After successful installation:

1. **Configure Environment**: Copy `.env.example` to `.env` and configure
2. **Setup Databases**: Initialize PostgreSQL and MongoDB
3. **Run Migrations**: `alembic upgrade head`
4. **Start Application**: See README.md for running the application

## Getting Help

If you encounter issues not covered here:
1. Check the [GitHub Issues](https://github.com/bloodbathwest-source/Music-AI-App/issues)
2. Review the error message carefully
3. Verify all system dependencies are installed
4. Try installing in a fresh virtual environment
