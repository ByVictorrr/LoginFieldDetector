import json
import os.path

import pytest
from login_field_detector import LoginFieldDetector, fetch_html


@pytest.fixture(scope="session")
def detector():
    """Fixture to initialize LoginFieldDetector."""
    file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "training_urls.json")
    with open(file_path, "r") as file:
        training_urls = json.load(file)
    detector = LoginFieldDetector()
    detector.train(urls=training_urls)
    return detector


@pytest.mark.parametrize("url", [
    "https://www.facebook.com/login",
    "https://twitter.com/login",
    "https://www.instagram.com/accounts/login/",
])
def test_media_urls(detector, url):
    """Test LoginFieldDetector with a set of media URLs."""
    # Example media URLs (mock URLs or replace with real ones for testing)
    html_text = fetch_html(url)
    if not any(i for t_p in detector.feature_extractor.get_features(html_text) for i in t_p):
        pytest.fail(f"HTML Feature extractor failed to extract any login tokens.")

    predictions = detector.predict(html_text)
    if len(predictions) < 1:
        pytest.fail(f"LoginFieldDetector failed with media URLs")
