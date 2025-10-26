# GitHub Copilot Instructions for Music-AI-App

## Project Context

This is a Music-Evolving AI web application built with Streamlit. The application uses evolutionary algorithms and LSTM neural networks to generate music based on user-selected parameters (genre, key, mode, emotion).

### Technology Stack

- **Python 3.8-3.11**: Supports Python 3.8+ (CI/CD tests 3.8, 3.9, 3.10; devcontainer uses 3.11)
- **Streamlit**: Web framework for the interactive UI
- **PyTorch**: Neural network framework for LSTM models
- **Libraries**:
  - `midiutil`: MIDI file generation
  - `pydub`: Audio file conversion (MIDI to WAV)
  - `matplotlib`: Data visualization
  - Standard ML libraries: `torch`, `torch.nn`, `torch.optim`

### Project Structure

- `app.py`: Placeholder file for the main Streamlit application
- `code version one`: Working implementation of the music generation application
- `.github/workflows/pylint.yml`: CI/CD pipeline for code quality
- `.devcontainer/`: Development container configuration for Codespaces

## Coding Standards

### Python Style Guidelines

1. **Follow PEP 8**: Use standard Python style conventions
2. **Linting**: Code must pass `pylint` checks (configured in CI/CD)
3. **Code Quality**: Maintain clean, readable, and well-documented code
4. **Naming Conventions**:
   - Use `snake_case` for variables and functions
   - Use `PascalCase` for classes
   - Use descriptive names that convey purpose

### Code Organization

1. **Imports**: Group imports in standard order (standard library, third-party, local)
2. **Functions**: Keep functions focused and single-purpose
3. **Classes**: Use classes for complex state management (e.g., neural networks)
4. **Comments**: Add comments for complex logic, especially in ML algorithms

## Development Workflow

### Testing and Quality Assurance

1. **Linting**: All Python code must pass `pylint` before committing
   - Run: `pylint $(git ls-files '*.py')`
   - CI/CD runs on Python 3.8, 3.9, and 3.10 to ensure compatibility
   - Development environment uses Python 3.11
2. **Manual Testing**: Test the Streamlit app locally before submitting changes
   - Run: `streamlit run "code version one"` (current working implementation)
   - Note: `app.py` is currently a placeholder
   - Verify all user interactions work as expected

### Working with Streamlit

1. **UI Components**: Use Streamlit's built-in widgets (`st.selectbox`, `st.slider`, `st.button`)
2. **State Management**: Be mindful of Streamlit's rerun behavior
3. **Performance**: Use `@st.cache_data` or `@st.cache_resource` for expensive computations
4. **User Feedback**: Provide clear feedback during long operations (`st.spinner`, `st.progress`)

### Working with Machine Learning Components

1. **PyTorch Models**: Define models as classes inheriting from `nn.Module`
2. **Training**: Keep training loops simple and well-documented
3. **Data**: Ensure generated music data is properly formatted for MIDI/audio output
4. **Random Seeds**: Consider using seeds for reproducible results when debugging

## Best Practices for This Repository

### When Adding Features

1. **Preserve Existing Functionality**: Don't break the current music generation flow
2. **Test Audio Output**: Always verify MIDI and WAV file generation works
3. **UI Consistency**: Maintain the current UI structure and user experience
4. **Parameter Validation**: Validate user inputs before processing

### When Fixing Bugs

1. **Reproduce First**: Understand the bug by running the app
2. **Minimal Changes**: Make the smallest change necessary to fix the issue
3. **Test Edge Cases**: Consider unusual input combinations
4. **Check Dependencies**: Ensure all required libraries are available

### When Refactoring

1. **Incremental Changes**: Make small, testable changes
2. **Preserve Interfaces**: Keep function signatures stable
3. **Document Changes**: Update comments and docstrings
4. **Verify Output**: Ensure generated music quality is not degraded

## Common Tasks

### Adding a New Genre

1. Add the genre to the selectbox options in `code version one`
2. Define appropriate scale patterns if needed
3. Update the melody generation logic to handle the new genre
4. Test the complete flow from selection to audio output

### Modifying the Neural Network

1. Update the `MusicLSTM` class definition in `code version one`
2. Note: The LSTM implementation is currently simplified/placeholder
3. Ensure forward pass is correctly implemented
4. Update training logic if needed
5. Test that the model can be instantiated and used

### Changing Audio Output

1. Modify MIDI generation in the `MIDIFile` section
2. Update audio conversion parameters in the `AudioSegment` section
3. Test both MIDI and WAV download functionality
4. Verify audio quality in playback

## Security and Dependencies

1. **No Secrets in Code**: Never commit API keys, credentials, or sensitive data
2. **Dependency Updates**: Test thoroughly when updating library versions
3. **Input Validation**: Always validate and sanitize user inputs
4. **Error Handling**: Gracefully handle errors and provide user-friendly messages

## Notes for Copilot

- This is a creative application focused on music generation
- User experience is paramount - keep the interface simple and responsive
- The evolutionary algorithm and LSTM are simplified for demonstration
- Audio quality and MIDI correctness are key success criteria
- The devcontainer is configured for easy Codespaces deployment
- Pylint must pass in CI/CD for all Python versions (3.8, 3.9, 3.10)
