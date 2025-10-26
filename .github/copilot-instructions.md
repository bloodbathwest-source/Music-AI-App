# GitHub Copilot Instructions for Music-AI-App

## Project Overview

This is a Music-AI-App - a Streamlit web application that uses AI to generate music based on user preferences. The application allows users to:
- Select music parameters (genre, key, mode, emotion)
- Generate music using evolutionary algorithms and LSTM neural networks
- Visualize the generated music
- Download the output as MIDI and WAV files

## Technology Stack

- **Framework**: Streamlit (Python web application framework)
- **AI/ML**: PyTorch for neural networks
- **Music Generation**: midiutil for MIDI file creation
- **Audio Processing**: pydub for audio conversion
- **Visualization**: matplotlib for data visualization
- **Language**: Python 3.8+

## Project Structure

- `app.py` - Main Streamlit application entry point (currently simplified)
- `code version one` - Original full implementation with music generation logic
- `.github/workflows/pylint.yml` - CI/CD pipeline for code quality checks

## Development Workflow

### Setup

1. Install Python 3.8 or higher
2. Install dependencies:
   ```bash
   pip install streamlit torch midiutil pydub matplotlib
   ```
3. Run the application locally:
   ```bash
   streamlit run app.py
   ```

### Testing and Linting

- **Linting**: The project uses pylint for code quality checks
  ```bash
  pylint $(git ls-files '*.py')
  ```
- The CI/CD pipeline runs pylint on every push for Python 3.8, 3.9, and 3.10
- Note: The current CI workflow only installs pylint and not project dependencies, which may cause import-related linting warnings

### Code Quality Standards

- Follow PEP 8 Python style guidelines
- All Python files should pass pylint checks
- Keep functions focused and modular
- Add docstrings for complex functions
- Use type hints where appropriate

## Coding Conventions

- Use 4 spaces for indentation (not tabs)
- Keep line length under 100 characters where reasonable
- Use descriptive variable names (e.g., `melody_notes` not `mn`)
- Organize imports in standard order: standard library, third-party, local
- Add comments for complex algorithms (especially music generation logic)

## Key Components to Understand

### Music Generation
- Uses evolutionary algorithms to generate melodies
- LSTM neural networks for pattern learning
- Supports multiple music scales (major, minor - note: dorian is in UI but not implemented)
- Generates MIDI files and converts them to WAV

### User Interface
- Streamlit-based web interface
- Input parameters: genre, key root, mode, emotion, generations
- Real-time music generation with progress indicator
- Audio playback and file download capabilities

## Common Tasks

### Adding New Features
- Keep the Streamlit interface simple and user-friendly
- Test music generation thoroughly with different parameters
- Ensure MIDI and audio export work correctly
- Update README.md if adding major functionality

### Fixing Bugs
- Check pylint output for code quality issues
- Test with various genre/key/mode combinations
- Verify audio playback in different browsers
- Ensure file downloads work properly

### Dependencies
- Be cautious when adding new dependencies
- Verify compatibility with Python 3.8+
- Update workflow if new dependencies require system packages
- Consider the impact on deployment and performance

## Best Practices for AI Development

- When modifying music generation algorithms, preserve the core logic
- Document parameters and their effects on music output
- Test with different random seeds for reproducibility
- Keep the evolutionary algorithm configurable via UI
- Ensure PyTorch models are properly initialized and trained

## File Organization

- Keep the main application logic in `app.py`
- Separate complex music generation logic into dedicated functions/modules if needed
- Store static assets (if any) in organized directories
- Don't commit generated files (MIDI, WAV) to the repository

## Security Considerations

- Don't expose sensitive API keys or credentials
- Sanitize user inputs before processing
- Be cautious with file uploads/downloads
- Validate all user-provided parameters

## Documentation

- Update README.md when adding major features
- Keep this copilot-instructions.md file up to date
- Add inline comments for complex music theory or algorithms
- Document any non-obvious dependencies or setup steps

## Notes for Copilot Coding Agent

- The current `app.py` is a simplified placeholder; the full implementation is in `code version one`
- When making changes, consider the Streamlit framework's reactive programming model
- Music generation can be CPU-intensive; consider performance implications
- Test both MIDI generation and WAV conversion when modifying audio logic
- The project is in early stages, so maintaining flexibility for future enhancements is important
