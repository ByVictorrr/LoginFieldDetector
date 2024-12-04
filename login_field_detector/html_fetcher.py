import logging
import asyncio
from datetime import datetime
from os import path, makedirs, cpu_count

from diskcache import Cache
from fake_useragent import UserAgent
from playwright.async_api import async_playwright, Page
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
        cache_dir = cache_dir or path.join(path.dirname(path.dirname(__file__)), "html_cache")
        makedirs(cache_dir, exist_ok=True)
        self.cache = Cache(cache_dir)
        self.failed_url_cache = Cache(f"{cache_dir}/failed_urls")
        self.screenshot_dir = path.join(cache_dir, "screenshots")
        makedirs(self.screenshot_dir, exist_ok=True)
        self.ttl = ttl
        self.max_concurrency = max_concurrency

    def fetch_html(self, url, force=False, screenshot=False):
        """Synchronously fetch a single URL."""
        result = self.fetch_all([url], force=force, screenshot=screenshot)
        return result.get(url, None)

    def fetch_all(self, urls: list, force: bool = False, screenshot: bool = False):
        """Fetch multiple URLs concurrently."""
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
        """Asynchronous batch fetcher."""
        tasks = [
            self._async_fetch_url(url, semaphore, screenshot)
            for url in urls if force or url not in self.cache
        ]
        return await asyncio.gather(*tasks)

    async def _async_fetch_url(self, url: str, semaphore: asyncio.Semaphore, screenshot: bool = False) -> tuple:
        """Fetch a single URL asynchronously."""
        async with semaphore:
            try:
                async with async_playwright() as p:
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
                            "--disable-http2",  # Add this flag
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

                    if not await self.navigate_with_retries(page, url):
                        return url, None

                    if not await self.handle_cloudflare_captcha(page, url, screenshot):
                        log.warning(f"Cloudflare CAPTCHA handling failed for {url}. Skipping...")
                        self.failed_url_cache.set(url, "captcha_failed", expire=self.ttl)
                        return url, None

                    await self.wait_for_page_ready(page)

                    html_content = await page.content()
                    log.debug(f"Successfully fetched content for {url}")
                    self.cache.set(url, html_content, expire=self.ttl)

                    if screenshot:
                        await self._save_screenshot(page, url, postfix="success")

                    return url, html_content

            except Exception as e:
                log.error(f"Error fetching {url}: {e}")
                await self.log_html_content(page, url, postfix="error")
                self.failed_url_cache.set(url, "unexpected_error", expire=self.ttl)
                return url, None

    async def navigate_with_retries(self, page: Page, url: str, retries: int = 3, backoff: int = 2) -> bool:
        """Navigate to a URL with retries and backoff."""
        for attempt in range(retries):
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                log.debug(f"Navigation successful for {url}")
                return True
            except Exception as e:
                log.warning(f"Retry {attempt + 1} for {url}: {e}")
                await asyncio.sleep(backoff ** attempt)  # Exponential backoff
        log.error(f"Failed to navigate to {url} after {retries} retries.")
        return False

    async def wait_for_page_ready(self, page: Page, timeout: int = 30000):
        """Generalized wait for page readiness."""
        try:
            await page.wait_for_load_state("networkidle", timeout=timeout)
            log.info("Network is idle.")

            await page.wait_for_function("document.readyState === 'complete'", timeout=timeout)
            log.info("Document is ready.")

            spinner_selectors = [".spinner", ".loading", ".progress", "[data-loading]", "[aria-busy='true']"]
            for selector in spinner_selectors:
                try:
                    await page.wait_for_selector(selector, state="detached", timeout=5000)
                    log.info(f"Spinner {selector} disappeared.")
                except Exception:
                    log.debug(f"No spinner or timeout waiting for {selector}.")
        except Exception as e:
            log.warning(f"Page readiness check failed: {e}")

    async def handle_cloudflare_captcha(self, page: Page, url: str, screenshot: bool) -> bool:
        """Enhanced Cloudflare CAPTCHA handling."""
        try:
            if screenshot:
                await self._save_screenshot(page, url, postfix="pre_cloudflare")
            await asyncio.sleep(3)
            content = await page.content()
            if "verify you are human" in content.lower():
                log.info(f"Cloudflare CAPTCHA detected for {url}. Attempting to handle...")
                captcha_frame = await page.query_selector("iframe[src*='/cdn-cgi/challenge-platform/']")
                if captcha_frame:
                    frame = await captcha_frame.content_frame()
                    checkbox = await frame.query_selector("input[type='checkbox']")
                    if checkbox:
                        await checkbox.click()
                        await asyncio.sleep(5)  # Allow processing
                        if screenshot:
                            await self._save_screenshot(page, url, postfix="post_cloudflare")
                        return True
            return True
        except Exception as e:
            log.error(f"Error handling Cloudflare CAPTCHA for {url}: {e}")
            return False

    async def _save_screenshot(self, page: Page, url: str, postfix=None):
        """Save a screenshot of the page."""
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            filename = path.join(
                self.screenshot_dir,
                f"{url.replace('/', '_').replace(':', '_')}_{timestamp}{f'_{postfix}' if postfix else ''}.png"
            )
            await page.screenshot(path=filename)
            log.debug(f"Screenshot saved: {filename}")
        except Exception as e:
            log.warning(f"Failed to take screenshot for {url}: {e}")

    async def log_html_content(self, page: Page, url: str, postfix=None):
        """Log and save HTML content for debugging."""
        try:
            content = await page.content()
            filename = path.join(
                self.screenshot_dir,
                f"{url.replace('/', '_').replace(':', '_')}_{postfix or 'content'}.html"
            )
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)
            log.debug(f"HTML content saved: {filename}")
        except Exception as e:
            log.warning(f"Failed to save HTML content for {url}: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fetcher = HTMLFetcher(max_concurrency=3)
    _urls = ["https://example.com", "https://x.com/i/flow/login"]
    res = fetcher.fetch_all(_urls, force=True, screenshot=True)
    for _url, html_text in res.items():
        print(f"Fetched {len(html_text)} characters from {_url}" if html_text else f"Failed to fetch {_url}")
