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
        """
        HTMLFetcher for downloading HTML content with caching, retry, and stealth support.

        :param cache_dir: Directory for persistent cache storage.
        :param ttl: Time-to-live (in seconds) for the cache.
        :param max_workers: Number of concurrent threads for fetching.
        :param max_concurrency: Maximum number of concurrent Playwright tasks.
        """
        cache_dir = cache_dir if cache_dir else os.path.join(os.path.dirname(os.path.dirname(__file__)), "html_cache")
        os.makedirs(cache_dir, exist_ok=True)
        self.cache = Cache(cache_dir)
        self.failed_url_cache = Cache(f"{cache_dir}/failed_urls")
        self.screenshot_dir = os.path.join(cache_dir, "screenshots")
        os.makedirs(self.screenshot_dir, exist_ok=True)
        self.ttl = ttl
        self.max_workers = max_workers
        self.max_concurrency = max_concurrency

    async def _save_screenshot(self, page, url, prefix="failed"):
        """Save a screenshot of the page for debugging."""
        filename = os.path.join(
            self.screenshot_dir,
            f"{prefix}_{url.replace('/', '_').replace(':', '_')}.png"
        )
        await page.screenshot(path=filename)
        log.info(f"Screenshot saved: {filename}")

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((PlaywrightTimeoutError, Exception)),
        reraise=True
    )
    async def _fetch_page_content(self, url, page):
        """Fetch page content with retries and ensure the page is fully loaded."""
        log.info(f"Fetching: {url}")
        await page.goto(url, wait_until="domcontentloaded", timeout=90000)
        await asyncio.sleep(5)
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

                    # Launch browser
                    args = [
                        # "--disable-blink-features=AutomationControlled",  # Prevent detection of automation
                        # "--disable-http2",
                        # "--ignore-certificate-errors",
                        # "--disable-gpu",
                        # "--no-sandbox",
                        "--disable-extensions",  # Disable all extensions
                        "--disable-extensions-except",  # Ensure no extensions interfere
                        "--disable-component-extensions-with-background-pages",
                    ]
                    browser = await p.chromium.launch(
                        headless=False,
                    )
                    # Create an incognito browser context
                    context = await browser.new_context(
                        java_script_enabled=True,
                        permissions=["geolocation", "notifications"],  # Allow key permissions
                        user_agent= "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                                    "Chrome/114.0.0.0 Safari/537.36"
                        # viewport={"width": 1280, "height": 800},
                        # record_har_path=None,  # Ensure no data is stored
                    )
                    # Clear cookies before starting
                    await context.clear_cookies()
                    page = await context.new_page()
                    # Add custom headers
                    # await page.set_extra_http_headers({
                    # "accept-language": "en-US,en;q=0.9",
                    # "upgrade-insecure-requests": "1",
                    # "cache-control": "max-age=0"
                    # })

                    await stealth_async(page)

                    # Fetch page content with retries
                    html_content = await self._fetch_page_content(url, page)

                    # Take a screenshot if enabled
                    if screenshot:
                        await self._save_screenshot(page, url, prefix="success")

                    page.on("console", lambda msg: print(f"Console: {msg.type()} - {msg.text()}"))
                    page.on("request", lambda req: print(f"Request: {req.url}"))
                    page.on("response", lambda res: print(f"Response: {res.url} - {res.status}"))
                    print(await page.evaluate("navigator.webdriver"))  # Should return None
                    await browser.close()
                    self.cache.set(url, html_content, expire=self.ttl)
                    return url, html_content
            except Exception as e:
                log.error(f"Failed to fetch {url}: {e}")
                self.failed_url_cache.set(url, "failed", expire=self.ttl)

                # Save a screenshot on failure
                if screenshot:
                    await self._save_screenshot(page, url, prefix="failed")
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
