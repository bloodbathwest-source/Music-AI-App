# GitHub Copilot Coding Agent Guide

This guide helps you get the best results when using GitHub Copilot coding agent with the Music AI App repository.

## Table of Contents
- [Overview](#overview)
- [Creating Effective Issues for Copilot](#creating-effective-issues-for-copilot)
- [Repository Setup for Copilot](#repository-setup-for-copilot)
- [Best Practices](#best-practices)
- [Task Types Suitable for Copilot](#task-types-suitable-for-copilot)
- [Working with Copilot Pull Requests](#working-with-copilot-pull-requests)

## Overview

GitHub Copilot coding agent is an AI-powered developer that can autonomously work on issues in your repository. Think of it as a new team member that needs clear instructions and a well-organized codebase to be effective.

## Creating Effective Issues for Copilot

### Use Issue Templates

We provide three issue templates optimized for Copilot:

1. **Bug Report** (`bug_report.yml`) - For reporting and fixing bugs
2. **Feature Request** (`feature_request.yml`) - For new features
3. **Task for Copilot** (`task.yml`) - For focused, well-scoped tasks

### Essential Elements of a Good Issue

Every issue assigned to Copilot should include:

#### 1. Clear Description
```markdown
Add input validation to the music generation API endpoint to ensure 
tempo values are within acceptable range (60-200 BPM).
```

#### 2. Specific Files to Modify
```markdown
Files to modify:
- backend/app/routers/music.py
- tests/backend/test_music_api.py
```

#### 3. Acceptance Criteria
```markdown
Acceptance Criteria:
- [ ] Tempo validation rejects values < 60 or > 200
- [ ] Appropriate error message is returned (HTTP 400)
- [ ] Unit tests added for edge cases
- [ ] All existing tests pass
- [ ] API documentation updated
```

#### 4. Test Requirements
```markdown
Testing Requirements:
- Add test_invalid_tempo_too_low()
- Add test_invalid_tempo_too_high()
- Add test_valid_tempo_range()
- Verify error response format matches API standards
```

#### 5. Context & Constraints
```markdown
Context:
- Current validation only checks if tempo exists, not the value
- Similar validation pattern exists in backend/app/validators.py

Constraints:
- Must maintain backward compatibility
- No new dependencies
- Follow existing error handling patterns
```

## Repository Setup for Copilot

### Environment Configuration

Our repository is configured to work with Copilot coding agent:

1. **Dependencies**: All dependencies are specified in `requirements.txt` (Python) and `package.json` (JavaScript)
2. **Environment Variables**: Use `.env.example` files as templates
3. **CI/CD**: GitHub Actions workflows validate changes automatically

### Running the Project

Copilot can execute these commands to validate changes:

```bash
# Backend setup
pip install -r requirements.txt
cp .env.example .env
python run_backend.py

# Backend tests
pytest tests/backend/ -v
pytest tests/backend/ --cov=backend

# Linting
pylint backend/app/*.py --max-line-length=100

# Frontend setup
cd frontend
npm install
npm run dev

# Frontend tests
npm test
npm run lint
```

### Test Infrastructure

- **Backend**: Uses `pytest` with test files in `tests/backend/`
- **Coverage**: Aim for >80% test coverage
- **CI**: Tests run automatically on push and PR via GitHub Actions

## Best Practices

### ✅ Good Tasks for Copilot

Copilot excels at:

- **Bug fixes** with clear reproduction steps
- **Small features** (adding an API endpoint, new component)
- **Refactoring** specific functions or modules
- **Adding tests** for existing code
- **Documentation updates**
- **Code style improvements** (following linting rules)
- **Accessibility improvements** (ARIA labels, keyboard navigation)
- **Performance optimizations** for specific functions
- **Dependency updates** (when scope is clear)

### ❌ Tasks to Avoid Assigning to Copilot

- Complex architectural changes
- Multi-repository changes
- Tasks requiring domain expertise (ML model design)
- Sensitive security implementations
- Ambiguous requirements
- Changes requiring extensive human judgment

### Task Scoping Guidelines

**Too Broad:**
```markdown
Improve the music generation system
```

**Just Right:**
```markdown
Add caching to the music generation endpoint to avoid regenerating 
identical requests. Use Redis with TTL of 1 hour. Update 
backend/app/routers/music.py and add tests in 
tests/backend/test_music_cache.py
```

**Too Narrow:**
```markdown
Change line 42 in music.py from `tempo = 120` to `tempo = 130`
```
(Just make this change yourself!)

## Task Types Suitable for Copilot

### Bug Fixes
```markdown
**Description**: The /api/music/generate endpoint returns 500 error 
when tempo parameter is missing

**Files**: backend/app/routers/music.py

**Acceptance Criteria**:
- [ ] Return 400 error with descriptive message when tempo is missing
- [ ] Add test case for missing tempo parameter
- [ ] Update API documentation
```

### Feature Additions
```markdown
**Description**: Add ability to export generated music as MIDI file

**Files**:
- backend/app/routers/export.py
- backend/app/services/midi_export.py (new file)
- tests/backend/test_midi_export.py (new file)

**Acceptance Criteria**:
- [ ] New endpoint POST /api/export/midi
- [ ] Converts internal music format to MIDI
- [ ] Returns downloadable MIDI file
- [ ] Tests for valid and invalid inputs
- [ ] Documentation updated
```

### Refactoring
```markdown
**Description**: Extract duplicate validation logic into reusable validators

**Files**:
- backend/app/validators.py (new file)
- backend/app/routers/music.py
- backend/app/routers/export.py
- tests/backend/test_validators.py (new file)

**Acceptance Criteria**:
- [ ] Create validators.py with reusable validation functions
- [ ] Update routers to use new validators
- [ ] All existing tests pass
- [ ] No change in API behavior
```

### Documentation
```markdown
**Description**: Add API documentation for authentication endpoints

**Files**: 
- backend/app/routers/auth.py (add docstrings)
- README.md (add authentication section)

**Acceptance Criteria**:
- [ ] All auth endpoints have docstrings
- [ ] README includes authentication flow
- [ ] Example requests/responses provided
```

## Working with Copilot Pull Requests

### Review Process

1. **Automated Checks**: Wait for CI/CD to complete
   - Tests must pass
   - Linting must pass
   - Coverage requirements met

2. **Code Review**: Review Copilot's changes like any PR
   - Check that acceptance criteria are met
   - Verify tests are comprehensive
   - Ensure code follows project patterns

3. **Request Changes**: Use `@copilot` in comments
   ```markdown
   @copilot The validation should also check for negative tempo values.
   Please add this check and a corresponding test.
   ```

4. **Iterate**: Copilot can make additional commits based on feedback

### Common Feedback Examples

**Request Additional Tests:**
```markdown
@copilot Please add test cases for:
- Empty string tempo
- Tempo as a decimal number
- Tempo with special characters
```

**Request Documentation:**
```markdown
@copilot Please add docstrings to the new validation functions 
following the Google docstring format used elsewhere in the codebase.
```

**Point Out Issues:**
```markdown
@copilot The change in line 45 breaks backward compatibility. 
Please modify to handle both old and new parameter formats.
```

## Code Style & Conventions

### Python (Backend)

- Follow PEP 8 guidelines
- Use type hints: `def validate_tempo(tempo: int) -> bool:`
- Maximum line length: 100 characters
- Docstrings for all public functions (Google style)
- Use async/await for I/O operations

Example:
```python
async def generate_music(
    genre: str,
    tempo: int,
    duration: int
) -> MusicTrack:
    """
    Generate AI music with specified parameters.
    
    Args:
        genre: Music genre (pop, jazz, classical, etc.)
        tempo: Beats per minute (60-200)
        duration: Track duration in seconds (30-180)
        
    Returns:
        Generated music track object
        
    Raises:
        ValueError: If parameters are invalid
    """
    # Implementation
```

### JavaScript/React (Frontend)

- Use ES6+ features
- Functional components with hooks
- TypeScript types where applicable
- ESLint configuration provided

### Testing

- Test file naming: `test_*.py` for backend, `*.test.ts(x)` for frontend
- One test class per file being tested
- Test both success and error cases
- Use descriptive test names: `test_tempo_validation_rejects_negative_values()`

## CI/CD Integration

Our GitHub Actions workflows will automatically:

1. **Install dependencies** from requirements.txt / package.json
2. **Run tests** with pytest / npm test
3. **Check code style** with pylint / ESLint
4. **Report coverage** (target: >80%)

Copilot operates within this same CI/CD environment, so if the workflows work for humans, they'll work for Copilot.

## Troubleshooting

### Copilot is stuck or not making progress

- Check if acceptance criteria are too vague
- Ensure files to modify are clearly specified
- Verify the task is appropriately scoped (not too broad)
- Check if required context/documentation is missing

### Copilot's solution doesn't match expectations

- Provide more specific acceptance criteria
- Add examples of desired behavior
- Reference similar patterns in the codebase
- Use `@copilot` to request specific changes in PR comments

### Tests are failing

- Ensure test infrastructure works locally
- Check that test requirements are clear
- Verify environment setup is documented
- Make sure all dependencies are in requirements.txt

## Resources

- [GitHub Copilot Best Practices](https://docs.github.com/en/copilot/tutorials/coding-agent/get-the-best-results)
- [Onboarding Copilot Agent](https://github.blog/ai-and-ml/github-copilot/onboarding-your-ai-peer-programmer-setting-up-github-copilot-coding-agent-for-success/)
- [Project README](README.md)
- [Contributing Guidelines](CONTRIBUTING.md)

## Questions?

If you have questions about using Copilot with this repository:
1. Check this guide first
2. Review the [CONTRIBUTING.md](CONTRIBUTING.md)
3. Open a discussion on GitHub
4. Ask the maintainers

---

**Remember**: Copilot is a tool to help you, not replace you. Clear communication, well-defined tasks, and good repository practices will help you get the best results!
