# Dependency Management

This document describes the dependency management approach for the Music-AI-App project.

## Overview

The project uses a multi-file requirements approach to support different components (frontend, backend, and shared base dependencies).

## Requirements Files

### `requirements-base.txt`
Contains core dependencies shared across all components:
- Audio processing libraries (midiutil, pydub)
- Basic utilities (numpy, matplotlib)

### `requirements-backend.txt`
Contains backend-specific dependencies:
- FastAPI and related web framework components
- Machine learning frameworks (PyTorch, TensorFlow)
- Database and authentication libraries
- Testing tools

### `requirements-frontend.txt`
Contains frontend-specific dependencies:
- Streamlit web framework
- Machine learning frameworks for inference
- Visualization libraries

### `requirements.txt`
The complete requirements file used for full installations. This is auto-generated from `requirements.in` using `pip-compile`.

### `requirements.in`
High-level dependency specifications with flexible version constraints. Use this file to add or update dependencies, then regenerate `requirements.txt` with `pip-compile`.

## Version Constraints

### TensorFlow
- **Constraint**: `>=2.17.0,<2.21.0`
- **Rationale**: Allows installation of currently available TensorFlow versions while avoiding non-existent versions (e.g., 2.20.0)
- **Note**: TensorFlow versioning can be irregular; always check PyPI for available versions

### Pydantic
- **Constraint**: `pydantic==2.12.3` and `pydantic-settings==2.12.3`
- **Rationale**: Ensures compatibility between pydantic and pydantic-settings
- **Note**: These versions must match to avoid conflicts

### PyTorch
- **Constraint**: `>=2.0.0,<3.0`
- **Rationale**: Modern PyTorch 2.x with stable API

### Streamlit
- **Constraint**: `>=1.28.0,<2.0`
- **Rationale**: Modern Streamlit features while maintaining API stability

## Adding Dependencies

1. Add the dependency to `requirements.in` with a flexible version constraint
2. If it's component-specific, also add to the appropriate component file
3. Run `pip-compile requirements.in` to regenerate `requirements.txt`
4. Run `python validate_requirements.py` to validate all requirements files
5. Test installation: `pip install -r requirements.txt`

## Updating Dependencies

1. Update the version constraint in `requirements.in`
2. Run `pip-compile requirements.in --upgrade` to update pinned versions
3. Run validation and test installation
4. Update component-specific files if needed

## Validation

Use the `validate_requirements.py` script to check for:
- Invalid version constraints
- Non-existent package versions
- Duplicate packages with conflicting versions

Run: `python validate_requirements.py`

## CI/CD Integration

### Workflow Configurations

The project includes multiple CI/CD workflows that use component-specific requirements:

- **Backend tests**: Install from `requirements-backend.txt`
- **Frontend tests**: Install from `requirements-frontend.txt`
- **Full integration**: Install from `requirements.txt`

### Docker Builds

The `Dockerfile` provides a reproducible build environment:
- Python 3.10-slim base image
- System audio libraries (ffmpeg, libsndfile)
- All dependencies from `requirements.txt`

Build: `docker build -t music-ai-app .`

## Troubleshooting

### Dependency Conflicts

If you encounter dependency conflicts:
1. Check that pydantic and pydantic-settings versions match
2. Verify TensorFlow version is in range `2.17.0` to `2.20.x` (but not 2.20.0 exactly)
3. Run `pip check` to identify conflicts
4. Use `pip-compile` to resolve dependencies automatically

### Version Not Available

If a specific version isn't available on PyPI:
1. Check PyPI for available versions: `pip index versions <package>`
2. Update the version constraint in `requirements.in`
3. Regenerate `requirements.txt`
4. Run validation

### CI/CD Failures

If CI/CD fails during dependency installation:
1. Check the workflow logs for specific errors
2. Verify the requirements file being used
3. Test locally with the same Python version
4. Update version constraints if needed

## Best Practices

1. **Pin versions in requirements.txt**: Use exact versions for reproducibility
2. **Use ranges in requirements.in**: Allow flexibility for dependency resolution
3. **Test before committing**: Always validate and test installation
4. **Keep components separate**: Don't mix frontend and backend dependencies
5. **Document changes**: Update this file when making significant dependency changes
6. **Regular updates**: Periodically update dependencies for security patches

## References

- [pip-tools documentation](https://pip-tools.readthedocs.io/)
- [Python packaging guide](https://packaging.python.org/)
- [Dependency version specifications](https://peps.python.org/pep-0440/)
