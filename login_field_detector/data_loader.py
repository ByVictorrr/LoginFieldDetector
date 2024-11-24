import os
import time
import hashlib
import asyncio
from requests_html import AsyncHTMLSession, HTMLSession


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
    """A class to handle data fetching and caching."""

    def __init__(self, session=None):
        self.session = session or AsyncHTMLSession()
        self.session.cookies.set("CookieConsent", "true")

    async def fetch_html(self, url):
        """Fetch HTML content asynchronously with caching."""
        cache_file = DatasetCache.get_cache_file(url)
        # Check cache
        if DatasetCache.is_cache_valid(cache_file):
            print(f"Using cached HTML for {url}")
            with open(cache_file, "r", encoding="utf-8") as f:
                return f.read()

        # Fetch HTML asynchronously
        print(f"Fetching HTML for {url}")
        try:
            response = await self.session.get(url, timeout=10, allow_redirects=True)
            await response.html.arender(timeout=20)  # Render JavaScript if needed
            html = response.html.html

            # Save to cache
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(html)

            return html
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None

    async def fetch_all(self, urls):
        """Fetch HTML content for all URLs concurrently."""

        async def fetch_single(url):
            return await self.fetch_html(url)

        tasks = [fetch_single(url) for url in urls]
        results = await asyncio.gather(*tasks)

        # Filter out None results
        return [(html, url) for html, url in zip(results, urls) if html is not None]


def fetch_html(url):
    """Fetch HTML content asynchronously with caching."""
    cache_file = DatasetCache.get_cache_file(url)
    # Check cache
    if DatasetCache.is_cache_valid(cache_file):
        print(f"Using cached HTML for {url}")
        with open(cache_file, "r", encoding="utf-8") as f:
            return f.read()

    # Fetch HTML asynchronously
    print(f"Fetching HTML for {url}")
    try:
        with HTMLSession() as session:
            response = session.get(url, timeout=10, allow_redirects=True)
        response.html.render(timeout=20)  # Render JavaScript if needed
        html = response.html.html
        # Save to cache
        with open(cache_file, "w", encoding="utf-8") as f:
            f.write(html)

        return html
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None
