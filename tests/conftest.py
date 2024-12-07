import os
import zipfile
import pytest
import shutil


@pytest.fixture(scope="session", autouse=True)
def extract_html_files():
    """Extract HTML files from resources/tests_html.zip if not already extracted.

    Ensures valid and invalid HTML files are available for tests.
    """
    base_dir = os.path.dirname(__file__)
    archive_path = os.path.join(base_dir, "resources", "tests_html.zip")
    extract_path = os.path.join(base_dir, "feature_extraction")

    # Ensure the archive exists
    if not os.path.exists(archive_path):
        raise FileNotFoundError(f"{archive_path} is missing. Please add it to the resources folder.")

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
    print(f"Cleaning up extracted files from {extract_path}...")
    shutil.rmtree(extract_path, ignore_errors=True)

