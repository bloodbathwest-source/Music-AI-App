# Dependency Management with pip-tools

This project uses `pip-tools` to manage Python dependencies and ensure consistent, conflict-free installations across all environments.

## Overview

- **`requirements.in`**: Contains high-level dependencies with minimum version constraints
- **`requirements.txt`**: Auto-generated file with all pinned dependencies (generated from `requirements.in`)

## Setup

### Installing pip-tools

```bash
pip install pip-tools
```

## Usage

### Adding a New Dependency

1. Add the dependency to `requirements.in`:
   ```
   new-package>=1.0.0
   ```

2. Regenerate `requirements.txt`:
   ```bash
   pip-compile requirements.in
   ```

3. Install the updated dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Updating Dependencies

To update all dependencies to their latest compatible versions:

```bash
pip-compile --upgrade requirements.in
```

To update a specific package:

```bash
pip-compile --upgrade-package package-name requirements.in
```

### Installing Dependencies

For development:

```bash
pip install -r requirements.txt
```

Using pip-sync (recommended, ensures exact match):

```bash
pip-sync requirements.txt
```

## CI/CD Integration

All GitHub Actions workflows have been updated to use the pinned `requirements.txt` file, ensuring:

- Consistent dependency versions across all test environments
- No dependency resolution conflicts
- Reproducible builds

## Current High-Level Dependencies

The `requirements.in` file contains:

- `streamlit>=1.28.0` - Web framework for the interactive UI
- `torch>=2.0.0` - Neural network framework for LSTM models
- `matplotlib>=3.7.0` - Data visualization
- `midiutil>=1.2.1` - MIDI file generation
- `pydub>=0.25.1` - Audio file conversion
- `tensorflow==2.20.0` - Machine learning framework

## Benefits

1. **Reproducible Environments**: Same dependency versions across development, testing, and production
2. **Conflict Resolution**: pip-tools resolves all dependency conflicts automatically
3. **Easy Updates**: Update dependencies safely with `--upgrade` flag
4. **Clear Separation**: High-level intent in `.in`, resolved versions in `.txt`

## Troubleshooting

### Dependency Conflicts

If you encounter conflicts when adding a new dependency:

1. Check if it conflicts with existing packages in `requirements.in`
2. Try compiling with verbose output:
   ```bash
   pip-compile -v requirements.in
   ```
3. Adjust version constraints in `requirements.in` if needed

### Installation Issues

If installation fails:

1. Ensure you're using the pinned `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

2. Clear pip cache if needed:
   ```bash
   pip cache purge
   pip install -r requirements.txt
   ```

## References

- [pip-tools Documentation](https://github.com/jazzband/pip-tools)
- [Python Packaging Guide](https://packaging.python.org/)
