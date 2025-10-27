#!/bin/bash
# Test script for validating requirements.txt installation
# This script should be run in a clean Python 3.12 environment

set -e  # Exit on error

echo "=========================================="
echo "Music AI App - Requirements Test Script"
echo "=========================================="
echo ""

# Check Python version
echo "1. Checking Python version..."
PYTHON_VERSION=$(python3 --version)
echo "   $PYTHON_VERSION"
if [[ ! "$PYTHON_VERSION" =~ "Python 3.1"[12] ]]; then
    echo "   ⚠️  Warning: Python 3.11 or 3.12 recommended"
fi
echo ""

# Check if in virtual environment
echo "2. Checking virtual environment..."
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "   ⚠️  Warning: Not in a virtual environment"
    echo "   Consider creating one: python3 -m venv venv && source venv/bin/activate"
else
    echo "   ✓ Virtual environment active: $VIRTUAL_ENV"
fi
echo ""

# Upgrade pip
echo "3. Upgrading pip..."
pip install --upgrade pip
echo ""

# Validate requirements.txt format
echo "4. Validating requirements.txt format..."
python3 << 'EOF'
import re
with open('requirements.txt', 'r') as f:
    content = f.read()
lines = [line.strip() for line in content.split('\n') if line.strip() and not line.strip().startswith('#')]
errors = []
for i, line in enumerate(lines, 1):
    if '==' in line:
        parts = line.split('==')
        if len(parts) != 2:
            errors.append(f"Line {i}: Invalid format - {line}")
if errors:
    print("❌ Validation errors:")
    for error in errors:
        print(f"  {error}")
    exit(1)
else:
    print(f"✓ Format valid ({len(lines)} packages)")
EOF
echo ""

# Install dependencies
echo "5. Installing dependencies from requirements.txt..."
echo "   This may take several minutes..."
pip install -r requirements.txt
echo ""

# Verify critical imports
echo "6. Verifying critical package imports..."
python3 << 'EOF'
import sys

packages_to_test = [
    ('fastapi', 'FastAPI'),
    ('uvicorn', 'Uvicorn'),
    ('torch', 'PyTorch'),
    ('numpy', 'NumPy'),
    ('librosa', 'Librosa'),
    ('streamlit', 'Streamlit'),
    ('sqlalchemy', 'SQLAlchemy'),
    ('pydantic', 'Pydantic'),
    ('pytest', 'Pytest'),
]

all_ok = True
for module_name, display_name in packages_to_test:
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"   ✓ {display_name:15} {version}")
    except ImportError as e:
        print(f"   ❌ {display_name:15} FAILED: {e}")
        all_ok = False

if not all_ok:
    print("\n❌ Some packages failed to import")
    sys.exit(1)
else:
    print("\n✓ All critical packages imported successfully")
EOF
echo ""

# Check for dependency conflicts
echo "7. Checking for dependency conflicts..."
pip check
echo ""

# Test basic functionality
echo "8. Testing basic imports from project..."
python3 << 'EOF'
import sys
sys.path.insert(0, '.')

try:
    from backend.app.services.ai_service import MusicGenerationService
    print("   ✓ Backend AI service imports successfully")
except Exception as e:
    print(f"   ⚠️  Backend AI service import issue: {e}")
    print("   (This is OK if dependencies like databases aren't configured)")
EOF
echo ""

echo "=========================================="
echo "✓ Requirements installation test complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Set up environment variables (cp .env.example .env)"
echo "2. Configure databases (PostgreSQL, MongoDB)"
echo "3. Run migrations (alembic upgrade head)"
echo "4. Start the application"
echo ""
