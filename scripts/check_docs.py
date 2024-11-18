#!/usr/bin/env python3
import re
import sys
from pathlib import Path

def check_readme():
    content = Path('README.md').read_text()
    
    # Required sections
    required_sections = {
        'Introduction': r'##\s+Introduction',
        'Installation': r'##\s+Installation',
        'Usage': r'##\s+Usage',
        'Architecture': r'##\s+Architecture|System Architecture',
        'Contributing': r'##\s+Contributing',
        'Requirements': r'##\s+Requirements|System Requirements',
        'Examples': r'##\s+Examples|Quick Start',
        'Configuration': r'##\s+Configuration',
        'Troubleshooting': r'##\s+Troubleshooting',
    }
    
    # Content requirements
    content_requirements = {
        'Hardware Requirements': r'macOS.*\d+\.\d+\+',
        'Python Version': r'Python.*\d+\.\d+\+',
        'MLX Version': r'MLX.*\d+\.\d+\.\d+',
        'Installation Steps': r'pip install|conda install',
        'Usage Example': r'```python|```bash',
        'Network Requirements': r'network|connectivity',
        'License Info': r'License|MIT',
    }
    
    missing_sections = []
    missing_content = []
    
    for section, pattern in required_sections.items():
        if not re.search(pattern, content, re.IGNORECASE):
            missing_sections.append(section)
            
    for req, pattern in content_requirements.items():
        if not re.search(pattern, content, re.IGNORECASE):
            missing_content.append(req)
    
    if missing_sections or missing_content:
        print("Documentation verification failed!")
        if missing_sections:
            print("Missing sections:", ", ".join(missing_sections))
        if missing_content:
            print("Missing content:", ", ".join(missing_content))
        sys.exit(1)
    
    # Check examples directory
    examples_dir = Path('examples')
    if not examples_dir.exists() or not list(examples_dir.glob('*.py')):
        print("Missing or empty examples directory")
        sys.exit(1)
    
    # Check API documentation
    api_docs = Path('docs/api')
    if not api_docs.exists() or not list(api_docs.glob('*.md')):
        print("Missing API documentation")
        sys.exit(1)
    
    print("Documentation verification passed!")
    sys.exit(0)

if __name__ == '__main__':
    check_readme()
