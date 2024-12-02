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

    async def _async_fetch_url(self, url, semaphore, proxy=None, screenshot=False):
        """Comprehensive URL fetching with stealth, CAPTCHA handling, dynamic content support, and security
        fallbacks."""
        async with semaphore:
            if url in self.failed_url_cache:
                log.warning(f"Skipping previously failed URL: {url}")
                return url, None

            try:
                async with async_playwright() as p:
                    # Launch browser with enhanced arguments and optional proxy
                    browser = await p.chromium.launch(
                        headless=True,
                        args=[
                            "--disable-blink-features=AutomationControlled",
                            "--disable-infobars",
                            "--disable-background-timer-throttling",
                            "--disable-renderer-backgrounding",
                            "--disable-dev-shm-usage",
                            "--no-sandbox",
                            "--disable-extensions",
                            "--disable-component-extensions-with-background-pages",
                            "--hide-scrollbars",
                            "--mute-audio",
                            "--disable-http2",
                        ],
                    )
                    context = await browser.new_context(
                        viewport={"width": 1920, "height": 1080},
                        locale="en-US",
                        user_agent=UserAgent().random,
                        proxy={"server": proxy} if proxy else None,
                        ignore_https_errors=True,
                        extra_http_headers={"accept-language": "en-US,en;q=0.9"},
                    )
                    page = await context.new_page()
                    await stealth_async(page)  # Apply stealth adjustments to bypass bot detection

                    log.info(f"Fetching: {url}")

                    # Navigate to the URL with retries
                    for attempt in range(3):
                        try:
                            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                            log.debug(f"Navigation successful for {url}")
                            break
                        except Exception as e:
                            log.warning(f"Retry {attempt + 1} for {url}: {e}")
                            if attempt == 2:  # Final attempt failed
                                log.error(f"Navigation failed after 3 retries for {url}: {e}")
                                self.failed_url_cache.set(url, "navigation_error", expire=self.ttl)
                                return url, None

                    # Wait for network stability
                    try:
                        await page.wait_for_load_state("networkidle", timeout=15000)
                        log.debug(f"Network idle state achieved for {url}")
                    except Exception as e:
                        log.warning(f"Network idle state not achieved for {url}: {e}")

                    # Optional screenshot
                    if screenshot:
                        try:
                            await self._save_screenshot(page, url, prefix="success")
                        except Exception as e:
                            log.error(f"Failed to take screenshot for {url}: {e}")

                    # Detect and handle CAPTCHA (including Cloudflare checkbox CAPTCHA)
                    try:
                        captcha_frame = await page.query_selector("iframe[src*='captcha']")
                        if captcha_frame:
                            log.warning(f"CAPTCHA detected for {url}. Attempting to handle...")
                            frame = await captcha_frame.content_frame()
                            checkbox = await frame.query_selector("input[type='checkbox']")
                            if checkbox:
                                await checkbox.click()
                                log.debug(f"Cloudflare CAPTCHA checkbox clicked for {url}")
                                await asyncio.sleep(3)  # Allow Cloudflare to process the click
                            else:
                                log.warning(f"No checkbox found in CAPTCHA for {url}. Skipping...")
                                self.failed_url_cache.set(url, "captcha_detected", expire=self.ttl)
                                return url, None
                    except Exception as e:
                        log.warning(f"Error detecting CAPTCHA for {url}: {e}")

                    # Handle dynamic content stabilization
                    try:
                        last_height = await page.evaluate("document.body.scrollHeight")
                        for _ in range(10):  # Poll for layout stabilization
                            await asyncio.sleep(1)
                            new_height = await page.evaluate("document.body.scrollHeight")
                            if new_height == last_height:
                                log.debug(f"Page stabilized for {url}")
                                break
                            last_height = new_height
                        else:
                            log.warning(f"Page layout did not stabilize for {url}. Proceeding...")
                    except Exception as e:
                        log.warning(f"Error during dynamic content handling for {url}: {e}")

                    # Ensure the body element is visible
                    try:
                        await page.wait_for_selector("body", state="visible", timeout=5000)
                        log.debug(f"Body element visible for {url}")
                    except Exception as e:
                        log.warning(f"Body element not visible for {url}: {e}. Proceeding...")

                    # Fetch HTML content
                    try:
                        html_content = await page.content()
                        log.debug(f"Successfully fetched content for {url}")
                        self.cache.set(url, html_content, expire=self.ttl)
                    except Exception as e:
                        log.error(f"Error fetching content for {url}: {e}")
                        return url, None

                    return url, html_content

            except Exception as e:
                log.error(f"Unexpected error for {url}: {e}")
                self.failed_url_cache.set(url, "unexpected_error", expire=self.ttl)
                return url, None

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
