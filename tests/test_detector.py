import os
import json
import pytest
import asyncio
from login_field_detector import LoginFieldDetector, DataLoader, fetch_html


@pytest.fixture(scope="session")
def detector():
    """Synchronous fixture to initialize and train LoginFieldDetector."""

    async def async_setup():
        # Resolve `load_html` asynchronously
        data_loader = DataLoader()
        file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "training_urls.json")

        with open(file_path, "r") as file:
            training_urls = json.load(file)[:20]

        html_data = await data_loader.fetch_all(training_urls)

        # Initialize and train the detector
        detector = LoginFieldDetector()
        detector.train([html for html, _ in html_data])  # Pass only HTML data
        return detector

    # Run the async setup synchronously
    return asyncio.run(async_setup())


@pytest.mark.parametrize("url", [
    "https://www.facebook.com/login",
    "https://twitter.com/login",
    "https://www.instagram.com/accounts/login/",
])
def test_media_urls(detector, url):
    """Test LoginFieldDetector with a set of media URLs."""
    if not (html_text := fetch_html(url)):
        pytest.fail(f"Failed fetching {url}.")
    elif not any(i for t_p in detector.feature_extractor.get_features(html_text) for i in t_p):
        pytest.fail(f"HTML Feature extractor failed to extract any login tokens.")
    elif not detector.predict(html_text):
        pytest.fail(f"LoginFieldDetector failed with media URLs")

