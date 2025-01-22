import warnings
import json
from pathlib import Path
import re
import subprocess
from typing import List, Tuple, Dict
import nbformat
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from tabulate import tabulate
from tqdm.auto import tqdm
import logging
from collections import defaultdict
import argparse

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


@dataclass
class BrokenLink:
    """Data class to store information about broken links"""

    notebook_path: str
    url: str
    alt_text: str
    cell_number: int
    line_number: int
    status_code: str


class NotebookLinkChecker:
    def __init__(
        self, project_dir: str = ".", output_file: str = "broken_links_report.md"
    ):
        """Initialize with project directory path"""
        self.project_dir = Path(project_dir)
        self.output_file = Path(output_file)
        self.broken_links: List[BrokenLink] = []

    def find_notebooks(self) -> List[Path]:
        """Find all Jupyter notebooks in the project directory"""
        return list(self.project_dir.rglob("*.ipynb"))

    def extract_links_from_notebook(
        self, notebook: nbformat.NotebookNode
    ) -> Dict[str, List[Tuple[str, int, int]]]:
        """Extract links from a Jupyter notebook and store their occurrences."""
        extracted_links = defaultdict(list)
        markdown_pattern = r"\[([^\]]+)\]\(([^)\s]+)\)"

        for cell_idx, cell in enumerate(notebook["cells"], 1):
            if cell["cell_type"] in ("markdown", "code"):
                lines = cell["source"].split("\n")
                for line_idx, line in enumerate(lines, 1):
                    matches = re.findall(markdown_pattern, line)
                    for alt_text, url in matches:
                        parsed_url = urlparse(url)
                        if parsed_url.scheme in ("http", "https"):
                            extracted_links[url].append((alt_text, cell_idx, line_idx))

        return extracted_links

    def check_url(self, url: str) -> Tuple[str, str]:
        """Check if URL is accessible using curl"""
        try:
            # Use curl to check URL status with timeouts
            result = subprocess.run(
                ["curl", "-sI", "-m", "10", url],  # 10 second timeout
                capture_output=True,
                text=True,
                timeout=15,  # process timeout
            )

            # Extract status code from curl output
            status_line = result.stdout.split("\n")[0]
            status_code = (
                status_line.split(" ")[1]
                if len(status_line.split(" ")) > 1
                else "Error"
            )

            return url, status_code
        except subprocess.TimeoutExpired:
            return url, "Timeout"
        except Exception as e:
            return url, f"Error: {str(e)}"

    def process_notebook(self, notebook_path: Path):
        """Process a single notebook to find broken links"""
        try:
            # Read notebook content once
            notebook = nbformat.read(notebook_path, as_version=4)

            # Extract links from the notebook
            extracted_links = self.extract_links_from_notebook(notebook)

            # Check each URL in parallel
            with ThreadPoolExecutor(max_workers=5) as executor:
                results = executor.map(self.check_url, extracted_links.keys())

                for url, status_code in results:
                    # Check for 404, 5xx errors, timeouts and other errors
                    if status_code.startswith(("404", "5")) or status_code.startswith(
                        ("Error", "Timeout")
                    ):
                        relative_path = notebook_path.relative_to(self.project_dir)
                        self.broken_links.append(
                            BrokenLink(
                                str(relative_path),
                                url,
                                extracted_links[url][0][0],
                                extracted_links[url][0][1],
                                extracted_links[url][0][2],
                                status_code,
                            )
                        )

        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Error processing notebook {notebook_path}: {str(e)}")

    def generate_report(self) -> str:
        """Generate markdown report of broken links"""
        if not self.broken_links:
            logger.info("No broken links found!")
            return

        # Prepare data for tabulate
        table_data = [
            [
                link.notebook_path,
                link.url,
                link.alt_text,
                link.cell_number,
                link.line_number,
                link.status_code,
            ]
            for link in self.broken_links
        ]

        headers = [
            "Notebook Path",
            "Broken Link",
            "Alt Text",
            "Cell No.",
            "Line No.",
            "Status Code",
        ]
        self.output_file.write_text(
            tabulate(table_data, headers=headers, tablefmt="pipe")
        )

        logger.info(f"Report saved to {self.output_file}")

    def run(self):
        """Main method to run the link checker"""
        notebooks = self.find_notebooks()
        if not notebooks:
            logger.info("No Jupyter notebooks found in the project directory.")
            return

        logger.info(f"Found {len(notebooks)} notebooks. Checking links...")

        # Process each notebook
        for notebook in tqdm(notebooks, desc="Extracting broken links"):
            self.process_notebook(notebook)

        # Generate report
        self.generate_report()


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Check Jupyter notebooks for broken links."
    )
    parser.add_argument(
        "-p",
        "--project_dir",
        type=str,
        default=".",
        help="Path to the project directory containing Jupyter notebooks. Default is current directory.",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="broken_links_report.md",
        help="Path to the output markdown file for the broken links report. Default is 'broken_links_report.md'.",
    )

    # Parse arguments
    args = parser.parse_args()

    # Create and run the link checker with parsed arguments
    checker = NotebookLinkChecker(
        project_dir=args.project_dir, output_file=args.output_path
    )
    checker.run()
