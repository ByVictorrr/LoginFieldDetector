import os
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from playwright.sync_api import sync_playwright


class DatasetCache:
    CACHE_DIR = "html_cache"
    TTL_SECONDS = 24 * 3600  # Cache time-to-live: 24 hours

    @classmethod
    def get_cache_file(cls, url):
        """Generate a unique cache file name based on the URL."""
        if not os.path.exists(cls.CACHE_DIR):
            os.makedirs(cls.CACHE_DIR)

        hashed_url = hashlib.md5(url.encode()).hexdigest()
        return os.path.join(cls.CACHE_DIR, f"{hashed_url}.html")

    @classmethod
    def is_cache_valid(cls, cache_file):
        """Check if the cache file exists and is within the TTL."""
        if not os.path.exists(cls.CACHE_DIR):
            os.makedirs(cls.CACHE_DIR)
        if not os.path.exists(cache_file):
            return False
        file_age = time.time() - os.path.getmtime(cache_file)
        return file_age < cls.TTL_SECONDS


class DataLoader:
    """A class to handle data fetching and caching with Playwright."""

    def __init__(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=True)  # Set headless=False for debugging

    def fetch_html(self, url):
        """Fetch HTML content with caching."""
        cache_file = DatasetCache.get_cache_file(url)

        # Check cache
        if DatasetCache.is_cache_valid(cache_file):
            print(f"Using cached HTML for {url}")
            return cache_file

        # Fetch HTML using Playwright
        print(f"Fetching HTML for {url}")
        try:
            context = self.browser.new_context()
            page = context.new_page()
            page.goto(url, timeout=30000)  # Timeout in milliseconds
            page.wait_for_load_state("networkidle")  # Wait for the page to fully load
            html = page.content()  # Get the HTML content of the page
            page.close()

            # Save to cache
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(html)

            return cache_file
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None

    def fetch_all(self, urls, max_threads=5):
        """Fetch HTML content for all URLs using threading."""
        results = []

        def task(url):
            return url, self.fetch_html(url)

        # Use ThreadPoolExecutor for parallel fetching
        with ThreadPoolExecutor(max_threads) as executor:
            futures = [executor.submit(task, url) for url in urls]

            for future in as_completed(futures):
                url, cache_file = future.result()
                if cache_file:
                    results.append((cache_file, url))

        return results

    def close(self):
        """Clean up resources."""
        self.browser.close()
        self.playwright.stop()


# Example Usage
if __name__ == "__main__":
    urls = [
        "https://example.com",
        "https://another-example.com",
        "https://google.com"
    ]

    loader = DataLoader()
    try:
        results = loader.fetch_all(urls, max_threads=5)
        for file_path, url in results:
            print(f"Cached HTML for {url} at {file_path}")
    finally:
        loader.close()
