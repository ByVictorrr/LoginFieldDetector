import os
import time
import hashlib
from requests_html import HTMLSession
from playwright.sync_api import sync_playwright

# Configuration
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "html_cache")
TTL_SECONDS = 24 * 3600  # 24 hours

# Initialize the cache directory
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

SESSION = HTMLSession()  # Single session for all requests


def get_cache_file(url):
    """Generate a unique cache file name based on the URL."""
    hashed_url = hashlib.md5(url.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{hashed_url}.html")


def is_cache_valid(cache_file):
    """Check if the cache file exists and is within the TTL."""
    if not os.path.exists(cache_file):
        return False
    file_age = time.time() - os.path.getmtime(cache_file)
    return file_age < TTL_SECONDS


def fetch_html(url):
    """
    Fetch HTML content from a URL, including JavaScript-rendered content.
    Uses a cache to avoid redundant requests within the TTL.
    """
    cache_file = get_cache_file(url)

    # Use cache if valid
    if is_cache_valid(cache_file):
        print(f"Using cached HTML for {url}")
        with open(cache_file, "r", encoding="utf-8") as f:
            return f.read()

    # Fetch the HTML and cache it
    print(f"Fetching and caching HTML for {url}")
    try:
        response = SESSION.get(url, timeout=10, allow_redirects=True)
        response.html.render(timeout=20)  # Adjust timeout for JavaScript rendering
        html = response.html.html

        # Cache the HTML
        with open(cache_file, "w", encoding="utf-8") as f:
            f.write(html)

        return html
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None
