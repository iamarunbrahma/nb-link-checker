# Notebook Link Checker

This tool scans through all Jupyter notebooks in a specified directory, extracts URLs from markdown and code cells, and checks if the extracted links are valid or broken.

## Features

- Recursively finds all Jupyter notebooks in a project directory and extracts links from both markdown and code cells in each notebook
- Validates if the extracted links are valid or broken i.e. 404, 5xx, etc.
- Generates a detailed markdown report of broken links along with the cell number and line number in the notebook where the link is located.

## Installation

1. Install uv (if not already installed):
```bash
pip install uv
```

2. Clone the repository and set up the environment:
```bash
git clone [repository-url]
cd nb-link-checker
uv venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt  # For development
```

## Usage

Basic usage with default settings:
```bash
python nb_link_checker.py
```

Specify a custom project directory and output file:
```bash
python nb_link_checker.py -p /path/to/notebooks -o custom_report.md
```

### Command Line Arguments

- `-p, --project_dir`: Path to the project directory containing Jupyter notebooks (default: current directory)
- `-o, --output_path`: Path to the output markdown file for the broken links report (default: broken_links_report.md)
