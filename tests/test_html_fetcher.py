import json
import os
import pytest
from login_field_detector import HTMLFetcher

APP_DIR = os.path.dirname(os.path.dirname(__file__))


@pytest.fixture(scope="module")
def fetcher():
    """Fixture to initialize the HTMLFetcher."""
    return HTMLFetcher(cache_dir=os.path.join(APP_DIR, "test_cache"))


def test_valid_urls(fetcher):
    """Test handling of redirects."""
    with open(os.path.join(APP_DIR, "dataset", "failed_urls.json"), "r") as fp:
        training_urls = json.load(fp).keys()
    html_content_list = fetcher.fetch_all(training_urls, force=True, screenshot=True).values()
    assert len(html_content_list) >= .8 * len(training_urls), \
        f"Failed to fetch at least 80 percent of the training_urls"


def test_fetch_valid_url(fetcher):
    """Test fetching a valid URL."""
    url = "https://www.example.com"
    html_content = fetcher.fetch_html(url)
    assert html_content is not None, f"Failed to fetch HTML content from {url}"


def test_redirect_handling(fetcher):
    """Test handling of redirects."""
    url = "http://github.com"  # Redirects to https://github.com
    html_content = fetcher.fetch_html(url)
    assert html_content is not None, f"Failed to handle redirect for {url}"


def test_invalid_url(fetcher):
    """Test fetching an invalid URL."""
    url = "https://invalid.example.com"
    html_content = fetcher.fetch_html(url)
    assert html_content is None, f"Fetcher did not handle invalid URL {url} correctly"


def test_timeout_handling(fetcher):
    """Test handling of slow-loading pages."""
    url = "https://httpstat.us/200?sleep=10000"  # Deliberately slow page
    with pytest.raises(Exception):
        fetcher.fetch_html(url)


def test_malformed_url(fetcher):
    """Test handling of malformed URLs."""
    url = "htp:/malformed.url"
    with pytest.raises(Exception):
        fetcher.fetch_html(url)


def test_screenshot_functionality(fetcher):
    """Test screenshot capturing."""
    url = "https://www.example.com"
    fetcher.fetch_html(url, screenshot=True)

    # Check if screenshot was saved
    screenshot_path = f"{url.replace('/', '_').replace(':', '_')}_screenshot.png"
    assert os.path.exists(screenshot_path), "Screenshot was not saved"

    # Clean up
    os.remove(screenshot_path)


def test_fetch_cached_url(fetcher):
    """Test fetching a URL that is already cached."""
    url = "https://www.example.com"
    fetcher.fetch_html(url)  # First fetch to cache it
    cached_content = fetcher.fetch_html(url)
    assert cached_content is not None, "Failed to fetch cached HTML content"


def test_force_refresh(fetcher):
    """Test force refresh to bypass cache."""
    url = "https://www.example.com"
    html_content = fetcher.fetch_html(url, force=True)
    assert html_content is not None, "Failed to force refresh HTML content"


def test_multiple_urls(fetcher):
    """Test fetching multiple URLs concurrently."""
    urls = [
        "https://www.example.com",
        "https://www.github.com",
        "https://www.python.org"
    ]
    results = fetcher.fetch_all(urls)
    assert len(results) == len(urls), "Mismatch in number of URLs fetched"
    for url, html_content in results.items():
        assert html_content is not None, f"Failed to fetch content for {url}"
