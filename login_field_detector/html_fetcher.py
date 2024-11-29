import logging
import os.path
from diskcache import Cache
import asyncio
from fake_useragent import UserAgent
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from playwright.async_api import TimeoutError as PlaywrightTimeoutError, async_playwright
from playwright_stealth import stealth_async

log = logging.getLogger(__name__)


class HTMLFetcher:
    def __init__(self, cache_dir=None, ttl=7 * 24 * 3600, max_workers=10, max_concurrency=5):
        """HTMLFetcher for downloading HTML content with caching and retry support.

        :param cache_dir: Directory for persistent cache storage.
        :param ttl: Time-to-live (in seconds) for the cache.
        :param max_workers: Number of concurrent threads for fetching.
        :param max_concurrency: Maximum number of concurrent Playwright tasks.
        """
        cache_dir = cache_dir if cache_dir else os.path.join(os.path.dirname(os.path.dirname(__file__)), "html_cache")
        self.cache = Cache(cache_dir)
        self.failed_url_cache = Cache(f"{cache_dir}/failed_urls")
        self.screenshot_dir = os.path.join(cache_dir, "screenshots")
        os.makedirs(self.screenshot_dir, exist_ok=True)
        self.ttl = ttl
        self.max_workers = max_workers
        self.max_concurrency = max_concurrency

    @retry(
        stop=stop_after_attempt(3),  # Retry up to 3 attempts
        wait=wait_exponential(multiplier=1, min=2, max=10),  # Exponential backoff between retries
        retry=retry_if_exception_type((PlaywrightTimeoutError, Exception)),  # Retry on specific exceptions
        reraise=True  # Raise the last exception if all retries fail
    )
    async def _fetch_page_content(self, url, page):
        """Helper method to fetch page content with retries."""
        log.info(f"Fetching: {url}")
        await page.goto(url, wait_until="domcontentloaded", timeout=90000)
        await page.wait_for_load_state("networkidle", timeout=60000)

        # Ensure the page is ready
        if not await page.is_visible("body"):
            raise Exception(f"Selector 'body' not visible for {url}")

        # Fetch content
        raw_content = await page.content()
        return raw_content.encode('utf-8', errors='replace').decode('utf-8', errors='replace')

    async def _async_fetch_url(self, url, semaphore, screenshot=False):
        """Fetch a single URL asynchronously using Playwright with stealth and retries."""
        async with semaphore:
            if url in self.failed_url_cache:
                log.warning(f"Skipping previously failed URL: {url}")
                return url, None
            if url in self.cache:
                log.info(f"Using cached HTML for {url}")
                return url, self.cache[url]

            try:
                async with async_playwright() as p:
                    # Generate a random User-Agent
                    ua = UserAgent()
                    selected_user_agent = ua.random

                    # Launch browser
                    browser = await p.chromium.launch(
                        headless=True,
                        args=["--disable-http2", "--ignore-certificate-errors"]
                    )
                    context = await browser.new_context(
                        user_agent=selected_user_agent,
                        viewport={"width": 1280, "height": 800}
                    )
                    page = await context.new_page()

                    # Apply stealth settings
                    await stealth_async(page)

                    # Fetch page content with retries
                    html_content = await self._fetch_page_content(url, page)

                    # Take a screenshot if the flag is enabled
                    if screenshot:
                        screenshot_path = os.path.join(
                            self.screenshot_dir,
                            f"{url.replace('/', '_').replace(':', '_')}_screenshot.png"
                        )
                        await page.screenshot(path=screenshot_path)
                        log.info(f"Screenshot saved for {url} at {screenshot_path}")

                    await browser.close()
                    self.cache.set(url, html_content, expire=self.ttl)
                    return url, html_content
            except Exception as e:
                log.error(f"Failed to fetch {url}: {e}")
                self.failed_url_cache.set(url, "failed", expire=self.ttl)
                return url, None

    async def _async_fetch_urls(self, urls, screenshot=False):
        """Fetch multiple URLs concurrently using Playwright."""
        semaphore = asyncio.Semaphore(self.max_concurrency)
        tasks = [self._async_fetch_url(url, semaphore, screenshot=screenshot) for url in urls]
        results = {}
        for url, html in await asyncio.gather(*tasks):
            if html:
                results[url] = html
        return results

    def fetch_all(self, urls, force=False, screenshot=False):
        """
        Synchronously fetch multiple URLs.

        :param urls: List of URLs to fetch.
        :param force: Force fetching even if URLs are cached or marked as failed.
        :param screenshot: Whether to take screenshots of the pages.
        :return: A dictionary mapping URLs to their HTML content.
        """
        log.info("Fetching all URLs...")
        # If force is enabled, clear failed cache for these URLs
        if force:
            for url in urls:
                if url in self.failed_url_cache:
                    log.info(f"Deleting {url} from failed_url_cache")
                    self.failed_url_cache.delete(url)
                if url in self.cache:
                    log.info(f"Deleting {url} from cache")
                    self.cache.delete(url)

        # Call the asynchronous fetcher synchronously
        return asyncio.run(self._async_fetch_urls(urls, screenshot=screenshot))

    def fetch_html(self, url, force=False, screenshot=False):
        """
        Synchronously fetch a single URL.

        :param url: URL to fetch.
        :param force: Force fetching even if URLs are cached or marked as failed.
        :param screenshot: Whether to take a screenshot of the page.
        :return: HTML content as a string or None if failed.
        """
        result = self.fetch_all([url], force=force, screenshot=screenshot)
        return result.get(url, None)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    fetcher = HTMLFetcher(max_workers=5, max_concurrency=3)

    # Example URLs
    _urls = [
        "https://example.com",
        "https://x.com/i/flow/login",
    ]

    # Fetch all URLs with screenshots
    _results = fetcher.fetch_all(_urls, force=True, screenshot=True)

    # Print results
    for _url, html_text in _results.items():
        if html_text:
            print(f"Fetched {len(html_text)} characters from {_url}")
        else:
            print(f"Failed to fetch {_url}")
