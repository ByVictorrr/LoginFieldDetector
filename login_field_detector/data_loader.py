import os
import time
import hashlib
import asyncio

from playwright.sync_api import sync_playwright
from playwright.async_api import async_playwright
from tqdm import tqdm


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
        if not os.path.exists(cache_file):
            return False
        file_age = time.time() - os.path.getmtime(cache_file)
        return file_age < cls.TTL_SECONDS


class AsyncDataLoader:
    """A class to handle data fetching and caching with Playwright Async API."""

    def __init__(self):
        self.browser = None
        self.playwright = None

    async def start(self):
        """Start Playwright and initialize the browser."""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)

    async def fetch_html(self, url):
        """Fetch HTML content with caching."""
        cache_file = DatasetCache.get_cache_file(url)
        await self.start()

        # Check cache
        if DatasetCache.is_cache_valid(cache_file):
            print(f"Using cached HTML for {url}")
            return cache_file

        # Fetch HTML using Playwright
        print(f"Fetching HTML for {url}")
        try:
            context = await self.browser.new_context()
            page = await context.new_page()
            await page.goto(url, timeout=30000)  # Timeout in milliseconds
            await page.wait_for_load_state("networkidle")  # Wait for the page to fully load
            html = await page.content()  # Get the HTML content of the page
            await page.close()

            # Save to cache
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(html)

            return cache_file
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None

    async def fetch_all(self, urls, max_concurrent=5):
        """Fetch HTML content for all URLs using asyncio.gather for concurrency."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_with_semaphore(url):
            async with semaphore:
                return await self.fetch_html(url), url

        # Use asyncio.gather to run tasks concurrently
        tasks = [fetch_with_semaphore(url) for url in urls]
        results = []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(urls), desc="Fetching URLs", unit="url"):
            try:
                result = await coro
                if result[1]:
                    results.append(result)
            except Exception as e:
                print(f"Error processing {result[0]}: {e}")
        return results

    async def close(self):
        """Clean up resources."""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

def fetch_html(url):
    """Fetch HTML content with caching."""
    cache_file = DatasetCache.get_cache_file(url)

    # Check cache
    if DatasetCache.is_cache_valid(cache_file):
        print(f"Using cached HTML for {url}")
        return cache_file

    # Fetch HTML using Playwright
    print(f"Fetching HTML for {url}")
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()
            page.goto(url, timeout=30000)  # Timeout in milliseconds
            page.wait_for_load_state("networkidle")  # Wait for the page to fully load
            html = page.content()  # Get the HTML content of the page
            page.close()
            browser.close()

        # Save to cache
        with open(cache_file, "w", encoding="utf-8") as f:
            f.write(html)

        return cache_file
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

# Example Usage
async def main():
    url_list = [
        "https://example.com",
        "https://another-example.com",
        "https://google.com",
    ]

    loader = AsyncDataLoader()
    try:
        await loader.start()
        results = await loader.fetch_all(url_list, max_concurrent=5)
        for file_p, url_text in results:
            print(f"Cached HTML for {url_text} at {file_p}")
    finally:
        await loader.close()

if __name__ == "__main__":
    asyncio.run(main())
