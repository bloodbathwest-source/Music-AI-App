# Contributing to Music AI App

Thank you for your interest in contributing to Music AI App! This document provides guidelines for contributing to the project.

## GitHub Copilot Coding Agent

This repository is optimized for GitHub Copilot coding agent. If you're using Copilot to work on issues, please refer to our [Copilot Guide](COPILOT_GUIDE.md) for best practices and tips on creating effective tasks.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/Music-AI-App.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests: `pytest tests/backend/`
6. Commit your changes: `git commit -m "Add some feature"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Create a Pull Request

## Development Setup

### Backend

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Run tests
pytest tests/backend/

# Start the development server
python run_backend.py
```

### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Set up environment variables
cp .env.local.example .env.local

# Start the development server
npm run dev
```

## Code Style

### Python
- Follow PEP 8 guidelines
- Use type hints where applicable
- Write docstrings for all functions and classes
- Maximum line length: 100 characters
- Run pylint before committing: `pylint backend/`

### JavaScript/React
- Use ES6+ features
- Follow React best practices
- Use functional components with hooks
- Run ESLint: `npm run lint`

## Testing

- Write tests for all new features
- Ensure all tests pass before submitting PR
- Aim for high test coverage (>80%)
- Test both success and failure cases

### Backend Tests
```bash
pytest tests/backend/ -v
pytest tests/backend/ --cov=backend
```

## Pull Request Guidelines

1. Update documentation if needed
2. Add tests for new features
3. Ensure all tests pass
4. Update CHANGELOG.md
5. Reference any related issues in the PR description
6. Keep PRs focused on a single feature or fix

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Accept constructive criticism gracefully
- Focus on what is best for the community

## Reporting Bugs

When reporting bugs, please include:
- Your operating system and version
- Python/Node.js version
- Steps to reproduce the bug
- Expected behavior
- Actual behavior
- Error messages or logs

## Feature Requests

We welcome feature requests! Please:
- Check if the feature has already been requested
- Clearly describe the feature and its use case
- Explain why it would be valuable to the project

## Questions?

Feel free to open an issue for questions or join our community discussions.

Thank you for contributing!
