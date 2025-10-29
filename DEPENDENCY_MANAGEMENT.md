# Dependency Management

This document explains how dependencies are managed in the Music-AI-App project.

## Overview

The project uses `pip-tools` to manage Python dependencies in a reliable and reproducible way:

- **`requirements.in`**: High-level dependencies (what the application needs)
- **`requirements.txt`**: Fully resolved dependencies (specific versions that work together)

## Key Changes

### Fixed Issues

- **Removed invalid dependency**: `tensorflow==2.20.0` was removed as this version doesn't exist and TensorFlow is not used in the codebase
- **Simplified dependencies**: Only the actual dependencies needed for the Streamlit frontend are included
- **Python compatibility**: Verified to work with Python 3.8 through 3.12

### Current Dependencies

The application requires:

1. **streamlit** - Web framework for the interactive UI
2. **torch** - PyTorch for LSTM neural network models
3. **matplotlib** - Data visualization for music analysis
4. **midiutil** - MIDI file generation
5. **pydub** - Audio file conversion (MIDI to WAV)

## Using pip-tools

### Installation

```bash
pip install pip-tools
```

### Updating Dependencies

1. Edit `requirements.in` to add, remove, or update high-level dependencies
2. Compile the requirements file:
   ```bash
   pip-compile requirements.in
   ```
3. Install the updated dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Why pip-tools?

- **Reproducibility**: Ensures everyone uses the same dependency versions
- **Conflict Resolution**: Automatically resolves version conflicts
- **Clarity**: Separates what you need (`requirements.in`) from how to get it (`requirements.txt`)
- **Security**: Makes it easier to update dependencies for security patches

## CI/CD Integration

The GitHub Actions workflows have been updated to:

- Test across Python versions 3.8, 3.9, 3.10, 3.11, and 3.12
- Install dependencies from `requirements.txt`
- Run the test suite with proper dependency resolution

## Troubleshooting

### Installation Fails

If `pip install -r requirements.txt` fails:

1. Make sure you have a compatible Python version (3.8-3.12)
2. Upgrade pip: `pip install --upgrade pip`
3. Try installing dependencies one by one to identify the problematic package
4. Check network connectivity to PyPI

### Dependency Conflicts

If you get dependency conflict errors:

1. Update `requirements.in` with compatible version constraints
2. Re-run `pip-compile requirements.in`
3. Test the installation: `pip install -r requirements.txt`

### Adding New Dependencies

When adding a new dependency:

1. Add it to `requirements.in` with a minimum version constraint
2. Run `pip-compile requirements.in` to update `requirements.txt`
3. Test that the application still works
4. Commit both `requirements.in` and `requirements.txt`

## References

- [pip-tools documentation](https://pip-tools.readthedocs.io/)
- [Python Packaging Guide](https://packaging.python.org/)
