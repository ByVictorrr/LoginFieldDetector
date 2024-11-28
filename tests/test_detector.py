import os
import pytest
from login_field_detector import LoginFieldDetector


@pytest.fixture(scope="session")
def detector():
    """Synchronous fixture to initialize and train LoginFieldDetector."""
    detector = LoginFieldDetector(model_dir="model")
    detector.train(force=True, output_dir="model")  # Pass only HTML data
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
