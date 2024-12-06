import os
import zipfile
import pytest


@pytest.fixture(scope="session", autouse=True)
def extract_html_files():
    """Extracts HTML files from tests_html.zip if not already extracted."""
    archive_path = "tests_html.zip"
    extract_path = "tests/feature_extraction"

    if not os.path.exists(os.path.join(extract_path, "valid")):
        print(f"Extracting {archive_path} to {extract_path}...")
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"Extraction complete: {extract_path}")
    else:
        print(f"HTML files already extracted in {extract_path}")
