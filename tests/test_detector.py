import os
import json
import pytest
from login_field_detector import LoginFieldDetector


@pytest.fixture(scope="session")
def detector():
    """Synchronous fixture to initialize and train LoginFieldDetector."""
    file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "training_urls.json")
    with open(file_path, "r") as file:
        training_urls = json.load(file)[:20]

    # Initialize and train the detector
    detector = LoginFieldDetector()
    detector.train(urls=training_urls)  # Pass only HTML data
    return detector


@pytest.mark.parametrize("url", [
    "https://www.facebook.com/login",
    "https://twitter.com/login",
    "https://www.instagram.com/accounts/login/",
])
def test_media_urls(detector, url):
    """Test LoginFieldDetector with a set of media URLs."""
    if not detector.predict(url=url):
        pytest.fail(f"LoginFieldDetector failed with media URLs")
    print("Pytest succeeded.")
