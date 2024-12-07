import os
import zipfile
import pytest
import shutil

test_dir = os.path.dirname(__file__)


@pytest.fixture(scope="session", autouse=True)
def extract_html_files():
    """
    Extracts HTML files from tests_html.zip if not already extracted.
    Ensures valid and invalid HTML files are available for tests.
    """
    archive_path = os.path.join(test_dir, "tests_html.zip")
    extract_path = os.path.join(test_dir, "feature_extraction")
    print(f"Archive path: {archive_path}")
    print(f"Extract path: {extract_path}")

    # Ensure the archive exists
    if not os.path.exists(archive_path):
        raise FileNotFoundError(f"{archive_path} is missing. Please add it to the project root.")

    # Extract files only if necessary
    valid_dir = os.path.join(extract_path, "valid")
    invalid_dir = os.path.join(extract_path, "invalid")

    if not os.path.exists(valid_dir) or not os.path.exists(invalid_dir):
        print(f"Extracting {archive_path} to {extract_path}...")
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"Extraction complete: {extract_path}")
    else:
        print(f"HTML files already extracted in {extract_path}")

    yield  # Allow tests to run

    # Optional cleanup after tests
    if os.path.exists(extract_path):
        print(f"Cleaning up extracted files from {extract_path}...")
        shutil.rmtree(extract_path)
