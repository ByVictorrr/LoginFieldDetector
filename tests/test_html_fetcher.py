import os
import pytest
from login_field_detector import HTMLFetcher


@pytest.fixture(scope="module")
def fetcher():
    """Fixture to initialize the HTMLFetcher."""
    return HTMLFetcher(cache_dir=os.path.join(os.path.dirname(__file__), "test_cache"))


@pytest.mark.external
def test_fetch_valid_url(fetcher):
    """Test fetching a valid URL."""
    url = "https://www.example.com"
    html_content = fetcher.fetch_html(url)
    assert html_content is not None, f"Failed to fetch HTML content from {url}"


@pytest.mark.external
def test_redirect_handling(fetcher):
    """Test handling of redirects."""
    url = "http://github.com"
    html_content = fetcher.fetch_html(url)
    assert html_content is not None, f"Failed to handle redirect for {url}"


@pytest.mark.external
def test_invalid_url(fetcher):
    """Test fetching an invalid URL."""
    url = "https://invalid.example.com"
    html_content = fetcher.fetch_html(url)
    assert html_content is None, f"Fetcher did not handle invalid URL {url} correctly"


@pytest.mark.external
def test_timeout_handling(fetcher):
    """Test handling of slow-loading pages."""
    url = "https://httpstat.us/200?sleep=10000"
    with pytest.raises(Exception):
        fetcher.fetch_html(url)


@pytest.mark.external
def test_fetch_cached_url(fetcher):
    """Test fetching a cached URL."""
    url = "https://www.example.com"
    fetcher.fetch_html(url)  # Cache it
    cached_content = fetcher.fetch_html(url, force=False)
    assert cached_content is not None, "Failed to fetch cached HTML content"
