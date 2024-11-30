import logging
from os import path, makedirs, cpu_count
from diskcache import Cache
import asyncio
from fake_useragent import UserAgent
from playwright.async_api import async_playwright
from playwright_stealth import stealth_async

log = logging.getLogger(__name__)


class HTMLFetcher:
    def __init__(self, cache_dir=None, ttl=7 * 24 * 3600, max_concurrency=cpu_count()):
        """
        HTMLFetcher for downloading HTML content with caching, retry, and stealth support.

        :param cache_dir: Directory for persistent cache storage.
        :param ttl: Time-to-live (in seconds) for the cache.
        :param max_concurrency: Maximum number of concurrent Playwright tasks.
        """
        cache_dir = cache_dir if cache_dir else path.join(path.dirname(path.dirname(__file__)), "html_cache")
        makedirs(cache_dir, exist_ok=True)
        self.cache = Cache(cache_dir)
        self.failed_url_cache = Cache(f"{cache_dir}/failed_urls")
        self.screenshot_dir = path.join(cache_dir, "screenshots")
        makedirs(self.screenshot_dir, exist_ok=True)
        self.ttl = ttl
        self.max_concurrency = max_concurrency

    async def _save_screenshot(self, page, url, prefix="debug"):
        """Save a screenshot of the page with a timeout."""
        try:
            filename = path.join(
                self.screenshot_dir,
                f"{prefix}_{url.replace('/', '_').replace(':', '_')}.png"
            )
            log.debug(f"Starting screenshot for {url}")
            await page.screenshot(path=filename, timeout=10000)  # 10 seconds
            log.debug(f"Completed screenshot for {url}")
        except Exception as e:
            log.warning(f"Failed to take screenshot for {url}: {e}")

    async def _async_fetch_url(self, url, semaphore, screenshot=False):
        async with semaphore:
            if url in self.failed_url_cache:
                log.warning(f"Skipping previously failed URL: {url}")
                return url, None

            try:
                async with async_playwright() as p:
                    browser = await p.chromium.launch(headless=True)
                    context = await browser.new_context(
                        viewport={"width": 1280, "height": 800},
                        locale="en-US",
                        user_agent=UserAgent().chrome,

                    )
                    page = await context.new_page()
                    await stealth_async(page)

                    # Fetch the page
                    log.info(f"Fetching: {url}")
                    await page.goto(url, wait_until="domcontentloaded", timeout=60000)
                    await page.wait_for_load_state("networkidle", timeout=50000)
                    # Wait for page to stabilize
                    try:
                        await page.wait_for_selector("body", state="visible", timeout=5000)
                    except Exception as e:
                        log.warning(f"Selector 'body' not visible for {url}: {e}. Proceeding...")

                    # Fetch HTML content
                    html_content = await page.content()
                    self.cache.set(url, html_content, expire=self.ttl)

                    # Take a screenshot if needed
                    if screenshot:
                        try:
                            await self._save_screenshot(page, url, prefix="success")
                        except Exception as e:
                            log.warning(f"Failed to take screenshot for {url}: {e}")

                    return html_content

            except Exception as e:
                log.warning(f"Error fetching {url}: {e}")
                self.failed_url_cache.set(url, "failed", expire=self.ttl)
                return None

    async def _async_fetch_urls(self, urls, screenshot=False):
        """Fetch multiple URLs concurrently using Playwright."""
        semaphore = asyncio.Semaphore(self.max_concurrency)
        tasks = [self._async_fetch_url(url, semaphore, screenshot=screenshot) for url in urls]

        results = {}
        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        for task_result, url in zip(task_results, urls):
            if isinstance(task_result, Exception):
                log.warning(f"Error fetching {url}: {task_result}")
                self.failed_url_cache.set(url, "error", expire=self.ttl)
            else:
                results[url] = task_result

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
        if force:
            for url in urls:
                if url in self.failed_url_cache:
                    log.info(f"Deleting {url} from failed_url_cache")
                    self.failed_url_cache.delete(url)
                if url in self.cache:
                    log.info(f"Deleting {url} from cache")
                    self.cache.delete(url)

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

    fetcher = HTMLFetcher(max_concurrency=3)

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
