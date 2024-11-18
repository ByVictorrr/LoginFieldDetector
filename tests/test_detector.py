import pytest
from login_field_detector import LoginFieldDetector, fetch_html


@pytest.fixture(scope="session")
def detector():
    """Fixture to initialize LoginFieldDetector."""
    detector = LoginFieldDetector()
    detector.train()
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
    predictions = detector.predict(html_text)
    if len(predictions) < 1:
        pytest.fail(f"LoginFieldDetector failed with media URLs")
