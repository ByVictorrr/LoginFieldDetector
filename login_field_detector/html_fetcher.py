import logging
import asyncio
from datetime import datetime
from os import path, makedirs, cpu_count

from diskcache import Cache
from fake_useragent import UserAgent
from playwright.async_api import async_playwright, Page
from playwright_stealth import stealth_async

log = logging.getLogger(__name__)


async def wait_for_page_stabilization(page: Page, timeout=10):
    """Wait for the page layout to stabilize."""
    try:
        last_height = await page.evaluate("document.body.scrollHeight")
        for _ in range(timeout):
            await asyncio.sleep(1)
            new_height = await page.evaluate("document.body.scrollHeight")
            if new_height == last_height:
                log.debug("Page stabilized.")
                return
            last_height = new_height
        log.warning("Page layout did not stabilize.")
    except Exception as e:
        log.warning(f"Error during page stabilization: {e}")


async def navigate_with_retries(page: Page, url: str, retries: int = 3) -> bool:
    """Navigate to a URL with retries."""
    for attempt in range(retries):
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            log.debug(f"Navigation successful for {url}")
            return True
        except Exception as e:
            log.warning(f"Retry {attempt + 1} for {url}: {e}")
    log.error(f"Failed to navigate to {url} after {retries} retries.")
    return False


class HTMLFetcher:
    def __init__(self, cache_dir=None, ttl=7 * 24 * 3600, max_concurrency=cpu_count()):
        """
        HTMLFetcher for downloading HTML content with caching, retry, and stealth support.

        :param cache_dir: Directory for persistent cache storage.
        :param ttl: Time-to-live (in seconds) for the cache.
        :param max_concurrency: Maximum number of concurrent Playwright tasks.
        """
        cache_dir = cache_dir or path.join(path.dirname(path.dirname(__file__)), "html_cache")
        makedirs(cache_dir, exist_ok=True)
        self.cache = Cache(cache_dir)
        self.failed_url_cache = Cache(f"{cache_dir}/failed_urls")
        self.screenshot_dir = path.join(cache_dir, "screenshots")
        makedirs(self.screenshot_dir, exist_ok=True)
        self.ttl = ttl
        self.max_concurrency = max_concurrency

    def fetch_html(self, url, force=False, screenshot=False):
        """Synchronously fetch a single URL.

        :param url: URL to fetch.
        :param force: Force fetching even if URLs are cached or marked as failed.
        :param screenshot: Whether to take a screenshot of the page.
        :return: HTML content as a string or None if failed.
        """
        result = self.fetch_all([url], force=force, screenshot=screenshot)
        return result.get(url, None)

    def fetch_all(self, urls: list, force: bool = False, screenshot: bool = False):
        """Fetch multiple URLs concurrently.

        :param urls: List of URLs to fetch.
        :param force: Force fetching even if URLs are cached or marked as failed.
        :param screenshot: Whether to take screenshots at key points.
        :return: A dictionary of URLs mapped to their HTML content or None if failed.
        """
        if force:
            for url in urls:
                if url in self.failed_url_cache:
                    log.info(f"Deleting {url} from failed_url_cache")
                    self.failed_url_cache.delete(url)
                if url in self.cache:
                    log.info(f"Deleting {url} from cache")
                    self.cache.delete(url)
        semaphore = asyncio.Semaphore(self.max_concurrency)
        results = asyncio.run(self._fetch_all_async(urls, semaphore, force, screenshot))
        return {url: html for url, html in results if html}

    async def _fetch_all_async(self, urls: list, semaphore: asyncio.Semaphore, force: bool, screenshot: bool):
        """Asynchronous batch fetcher.

        :param urls: List of URLs to fetch.
        :param semaphore: Semaphore to control concurrency.
        :param force: Force fetching even if URLs are cached or marked as failed.
        :param screenshot: Whether to take screenshots at key points.
        :return: A dictionary of URLs mapped to their HTML content or None if failed.
        """
        tasks = [
            self._async_fetch_url(url, semaphore, screenshot)
            for url in urls if force or url not in self.cache
        ]
        return await asyncio.gather(*tasks)

    async def _async_fetch_url(self, url: str, semaphore: asyncio.Semaphore, screenshot: bool = False) -> tuple:
        """Fetch a single URL asynchronously with stealth, CAPTCHA handling, and dynamic content support.

        :param url: The URL to fetch.
        :param semaphore: Semaphore to control concurrency.
        :param screenshot: Whether to take screenshots at key points.
        :return: A tuple of (URL, HTML content or None if failed).
        """
        async with semaphore:
            try:
                async with async_playwright() as p:
                    # Launch browser with arguments optimized for stealth
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
                        ],
                    )
                    context = await browser.new_context(
                        user_agent=UserAgent().random,
                        viewport={"width": 1920, "height": 1080},
                        ignore_https_errors=True,
                        extra_http_headers={"accept-language": "en-US,en;q=0.9"},
                    )
                    page = await context.new_page()
                    await stealth_async(page)

                    log.info(f"Fetching: {url}")

                    # Navigate to the URL with retries
                    if not await navigate_with_retries(page, url):
                        return url, None

                    # Handle Cloudflare checkbox CAPTCHA
                    await asyncio.sleep(3)
                    if not await self.handle_cloudflare_captcha(page, url, screenshot):
                        log.warning(f"Cloudflare CAPTCHA handling failed for {url}. Skipping...")
                        self.failed_url_cache.set(url, "captcha_failed", expire=self.ttl)
                        return url, None

                    # Wait for page stabilization
                    await wait_for_page_stabilization(page)
                    await asyncio.sleep(3)

                    # Ensure the body is visible
                    try:
                        await page.wait_for_selector("body", state="visible", timeout=5000)
                        log.debug(f"Body element visible for {url}")
                    except Exception as e:
                        log.warning(f"Body not visible for {url}: {e}. Proceeding...")

                    # Fetch the page content
                    await asyncio.sleep(3)
                    html_content = await page.content()
                    log.debug(f"Successfully fetched content for {url}")
                    self.cache.set(url, html_content, expire=self.ttl)

                    # Take a screenshot if required
                    if screenshot:
                        await self._save_screenshot(page, url, postfix="success")

                    return url, html_content

            except Exception as e:
                log.error(f"Error fetching {url}: {e}")
                self.failed_url_cache.set(url, "unexpected_error", expire=self.ttl)
                return url, None

    async def handle_cloudflare_captcha(self, page: Page, url: str, screenshot: bool) -> bool:
        """Handle simple Cloudflare CAPTCHA challenges (e.g., checkbox).

        :param page: The Playwright Page instance.
        :param url: The URL being processed.
        :param screenshot: Whether to take screenshots at key points.
        :return: True if the CAPTCHA was successfully handled, False otherwise.
        """
        try:
            if screenshot:
                await self._save_screenshot(page, url, postfix="pre_cloudflare")
            await asyncio.sleep(3)
            captcha_frame = await page.query_selector("iframe[src*='/cdn-cgi/challenge-platform/']")
            content = await page.content()
            if "verify you are human by completing the action below" in content.lower():
                log.info(f"Cloudflare CAPTCHA detected for {url}. Attempting to handle...")
                frame = await captcha_frame.content_frame()
                checkbox = await frame.query_selector("input[type='checkbox']")
                if checkbox:
                    await checkbox.click()
                    await asyncio.sleep(3)  # Allow CAPTCHA to process
                    if screenshot:
                        await self._save_screenshot(page, url, postfix="post_cloudflare")
                        return True
            else:
                return True
        except Exception as e:
            log.error(f"Error handling Cloudflare CAPTCHA for {url}: {e}")
            return False

    async def _save_screenshot(self, page: Page, url: str, postfix=None):
        """Save a screenshot of the page."""
        try:
            log_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            filename = path.join(
                self.screenshot_dir,
                f"{url.replace('/', '_').replace(':', '_')}_{log_timestamp}{f'_{postfix}' if postfix else ''}.png"
            )
            await page.screenshot(path=filename)
            log.debug(f"Screenshot saved: {filename}")
        except Exception as e:
            log.warning(f"Failed to take screenshot for {url}: {e}")


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
