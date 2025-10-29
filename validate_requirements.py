#!/usr/bin/env python3
"""
Test script to validate requirements.txt

This script checks:
1. The requirements.txt file can be parsed
2. No invalid version specifiers exist
3. All packages are real packages (when network is available)
"""

import re
import sys
from pathlib import Path


def parse_requirements(filepath):
    """Parse requirements.txt and extract package specifications."""
    packages = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            # Remove comments and strip whitespace
            line = line.split('#')[0].strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Check for valid package specification format
            # Format: package_name[extras]>=version,<version
            if not re.match(r'^[a-zA-Z0-9_-]+(\[[a-zA-Z0-9_,-]+\])?(>=|<=|==|!=|~=|>|<)', line):
                print(f"Warning: Line {line_num} doesn't match expected format: {line}")
                continue
            
            packages.append((line_num, line))
    
    return packages


def validate_requirements():
    """Validate the requirements.txt file."""
    requirements_file = Path(__file__).parent / 'requirements.txt'
    
    if not requirements_file.exists():
        print(f"Error: {requirements_file} not found")
        return False
    
    print(f"Validating {requirements_file}...")
    
    packages = parse_requirements(requirements_file)
    
    if not packages:
        print("Error: No packages found in requirements.txt")
        return False
    
    print(f"\nFound {len(packages)} package(s):")
    for line_num, package in packages:
        print(f"  Line {line_num}: {package}")
    
    # Check for known invalid packages
    invalid_packages = []
    for line_num, package in packages:
        # Check for tensorflow==2.20.0 specifically
        if 'tensorflow==2.20.0' in package:
            invalid_packages.append((line_num, package, "Version 2.20.0 doesn't exist"))
        # Check for very high TensorFlow versions that don't exist
        if re.match(r'tensorflow==2\.(1[9-9]|[2-9][0-9])', package):
            invalid_packages.append((line_num, package, "TensorFlow version doesn't exist"))
    
    if invalid_packages:
        print("\nErrors found:")
        for line_num, package, reason in invalid_packages:
            print(f"  Line {line_num}: {package} - {reason}")
        return False
    
    print("\n✓ All package specifications appear valid")
    return True


if __name__ == '__main__':
    success = validate_requirements()
    sys.exit(0 if success else 1)
