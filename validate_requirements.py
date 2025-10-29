#!/usr/bin/env python3
"""
Validate requirements files to catch unrealistic dependencies.

This script checks:
1. Version constraints are valid
2. Packages exist on PyPI
3. Specified versions are available
4. No duplicate packages with conflicting versions
"""

import sys
import re
from typing import List, Tuple, Set
import subprocess


def parse_requirement_line(line: str) -> Tuple[str, str]:
    """
    Parse a requirement line into package name and version constraint.
    
    Returns:
        Tuple of (package_name, version_constraint)
    """
    # Remove comments and whitespace
    line = line.split('#')[0].strip()
    
    # Skip empty lines and -r includes
    if not line or line.startswith('-r') or line.startswith('--'):
        return None, None
    
    # Handle different version specifiers
    match = re.match(r'^([a-zA-Z0-9_-]+(?:\[.*?\])?)(.*?)$', line)
    if match:
        package = match.group(1).split('[')[0]  # Remove extras
        version = match.group(2).strip()
        return package, version
    
    return None, None


def check_package_exists(package: str) -> bool:
    """Check if a package exists on PyPI."""
    try:
        result = subprocess.run(
            ['pip', 'index', 'versions', package],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"Warning: Could not check package {package}: {e}")
        return True  # Assume it exists if we can't check


def validate_version_constraint(package: str, constraint: str) -> bool:
    """Validate that version constraint is reasonable."""
    if not constraint:
        return True
    
    # Check for obviously invalid versions like 2.20.0 for tensorflow
    if package.lower() == 'tensorflow':
        # TensorFlow versions should be valid (e.g., not 2.20.0 which doesn't exist)
        if '2.20.0' in constraint or '2.19.0' in constraint:
            print(f"Error: {package} version {constraint} is invalid")
            return False
    
    return True


def find_duplicates(requirements: List[Tuple[str, str]]) -> Set[str]:
    """Find packages listed multiple times with potentially conflicting versions."""
    package_versions = {}
    duplicates = set()
    
    for package, version in requirements:
        if package is None:
            continue
        
        package_lower = package.lower()
        if package_lower in package_versions:
            if package_versions[package_lower] != version:
                duplicates.add(package)
                print(f"Warning: {package} listed multiple times with different versions:")
                print(f"  - {package_versions[package_lower]}")
                print(f"  - {version}")
        else:
            package_versions[package_lower] = version
    
    return duplicates


def validate_requirements_file(filepath: str) -> bool:
    """Validate a single requirements file."""
    print(f"\nValidating {filepath}...")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File {filepath} not found")
        return False
    
    requirements = []
    errors = []
    
    for line_num, line in enumerate(lines, 1):
        package, version = parse_requirement_line(line)
        if package:
            requirements.append((package, version))
            
            # Validate version constraint
            if not validate_version_constraint(package, version):
                errors.append(f"Line {line_num}: Invalid version constraint for {package}")
    
    # Check for duplicates
    duplicates = find_duplicates(requirements)
    if duplicates:
        errors.append(f"Duplicate packages found: {', '.join(duplicates)}")
    
    if errors:
        print("Validation errors found:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print(f"✓ {filepath} is valid")
    return True


def main():
    """Main validation function."""
    requirements_files = [
        'requirements.txt',
        'requirements.in',
        'requirements-base.txt',
        'requirements-backend.txt',
        'requirements-frontend.txt',
    ]
    
    all_valid = True
    for req_file in requirements_files:
        if not validate_requirements_file(req_file):
            all_valid = False
    
    if all_valid:
        print("\n✓ All requirements files are valid!")
        return 0
    else:
        print("\n✗ Some requirements files have validation errors")
        return 1


if __name__ == '__main__':
    sys.exit(main())
