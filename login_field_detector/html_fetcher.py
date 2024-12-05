import json
import re
import logging
import asyncio
import os
import random
from datetime import datetime

from diskcache import Cache
from fake_useragent import UserAgent
from playwright.async_api import async_playwright
from playwright_stealth import stealth_async

log = logging.getLogger(__name__)

with open(os.path.join(os.path.dirname(__file__), "keywords.json"), "r") as key_fp:
    SPINNER_PATTERNS = json.load(key_fp)["spinner_patterns"]

DEFAULT_TIMEOUT = 60  # seconds


def playwright_timeout(seconds):
    """Convert a given time in seconds to milliseconds.

    :param seconds: The number of seconds.
    :return: The equivalent time in milliseconds.
    """
    return seconds * 1000


async def _wait_for_dynamic_content(page, max_retries=10, interval=1000):
    """Waits for dynamic content to stop updating.

    :param page: Playwright page instance.
    :param max_retries: Maximum number of retries.
    :param interval: Time to wait between retries (in milliseconds).
    :return: True if content stabilizes, False otherwise.
    """
    previous_html = ""
    for _ in range(max_retries):
        current_html = await page.content()
        if current_html == previous_html:
            log.info("Page content stabilized.")
            return True
        previous_html = current_html
        await asyncio.sleep(interval / 1000)
    log.warning("Page content did not stabilize.")
    return False


async def _wait_for_spinners(page, regex_patterns, timeout):
    """Wait for spinners or loading indicators matching regex patterns to disappear.

    :param page: Playwright page instance.
    :param regex_patterns: List of regex patterns to match spinner attributes.
    :param timeout: Timeout in seconds.
    """
    try:
        # Combine regex patterns into a single string
        combined_pattern = '|'.join(regex_patterns)

        # Precompile the regex in Python for validation
        compiled_regex = re.compile(combined_pattern, re.IGNORECASE)

        # JavaScript code to check spinner visibility without unsafe-eval
        js_code = """
        (pattern) => {
            const regex = new RegExp(pattern, 'i');
            const elements = Array.from(document.querySelectorAll('*'));
            return elements.some(element => {
                const attributes = Array.from(element.attributes).map(attr => attr.value || '');
                return (
                    regex.test(element.className || '') ||
                    regex.test(element.id || '') ||
                    attributes.some(attr => regex.test(attr))
                ) &&
                getComputedStyle(element).visibility !== 'hidden' &&
                getComputedStyle(element).display !== 'none';
            });
        }
        """

        # Call JavaScript with the regex pattern passed as an argument
        await page.wait_for_function(
            js_code,
            arg=compiled_regex.pattern,
            timeout=playwright_timeout(timeout),
        )
        log.info("All spinners matching the expanded regex have disappeared.")
    except asyncio.TimeoutError:
        log.warning("Spinners did not disappear within the timeout.")
    except Exception as e:
        log.error(f"Error waiting for spinners: {e}")


async def wait_for_page_ready(page, timeout=15):
    """Wait for the page to be fully loaded and stable."""
    try:
        # Wait for network idle
        await page.wait_for_load_state("networkidle", timeout=playwright_timeout(timeout))
        log.info("Network is idle.")

        # Wait for DOM to be fully loaded
        await page.wait_for_function("document.readyState === 'complete'", timeout=playwright_timeout(timeout))
        log.info("DOM is ready.")

        # Wait for spinners matching regex patterns
        await _wait_for_spinners(page, SPINNER_PATTERNS, timeout)

        # Wait for dynamic content to stabilize
        if not await _wait_for_dynamic_content(page):
            log.warning("Dynamic content did not stabilize.")
    except Exception as e:
        log.error(f"Page readiness failed: {e}")
        raise


async def navigate_with_retries(page, url, retries=3, backoff=2, timeout=60):
    for attempt in range(retries):
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=playwright_timeout(timeout))
            return True
        except Exception as e:
            delay = backoff * (2 ** attempt) + random.uniform(0, 1)  # Add jitter
            log.warning(f"Retry {attempt + 1} for {url} failed: {e}. Retrying in {delay:.2f}s...")
            await asyncio.sleep(delay)
    return False


class HTMLFetcher:
    def __init__(
            self,
            cache_dir=None,
            ttl=7 * 24 * 3600,  # Cache expiration time in seconds
            max_concurrency=4,  # Number of concurrent browser contexts
            browser_executable_path=None,  # Custom browser executable
            browser_launch_kwargs=None,
            context_kwargs=None,
    ):
        # Initialization remains unchanged
        cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), "html_cache")
        os.makedirs(cache_dir, exist_ok=True)

        self.cache = Cache(cache_dir)
        self.failed_url_cache = Cache(f"{cache_dir}/failed_urls")
        self.screenshot_dir = os.path.join(cache_dir, "screenshots")
        os.makedirs(self.screenshot_dir, exist_ok=True)

        self.ttl = ttl
        self.max_concurrency = max_concurrency
        self.browser_executable_path = browser_executable_path

        # Browser launch arguments
        self.browser_launch_kwargs = browser_launch_kwargs or {
            "headless": True,
            "executable_path": self.browser_executable_path,
            "args": [
                "--disable-blink-features=AutomationControlled",
                "--disable-http2",
                "--disable-gpu",
                "--no-sandbox",
            ],
        }

        # Context arguments
        self.context_kwargs = context_kwargs or {
            "user_agent": UserAgent().random,
            "bypass_csp": True,
            "viewport": {"width": 1920, "height": 1080},
            "ignore_https_errors": True,
            "extra_http_headers": {"accept-language": "en-US,en;q=0.9"},
        }

    def fetch_html(self, url, force=False, screenshot=False):
        """
        Synchronously fetch a single URL.

        :param url: URL to fetch.
        :param force: Force fetching even if URLs are cached or marked as failed.
        :param screenshot: Whether to take a screenshot of the page.
        :return: HTML content as a string or None if failed.
        """
        results = self.fetch_all([url], force=force, screenshot=screenshot)
        return results.get(url)

    def fetch_all(self, urls, force=False, screenshot=False):
        """
        Synchronously fetch multiple URLs.

        :param urls: List of URLs to fetch.
        :param force: Force fetching even if URLs are cached or marked as failed.
        :param screenshot: Whether to take a screenshot of the pages.
        :return: Dictionary of {url: html} for successfully fetched URLs.
        """
        return asyncio.run(self._fetch_all(urls, force=force, screenshot=screenshot))

    async def _fetch_all(self, urls, force=False, screenshot=False):
        """
        Asynchronously fetch multiple URLs.

        :param urls: List of URLs to fetch.
        :param force: Force fetching even if URLs are cached or marked as failed.
        :param screenshot: Whether to take a screenshot of the pages.
        :return: Dictionary of {url: html} for successfully fetched URLs.
        """
        # Clean up cache if forced
        if force:
            await self._cleanup_cache(urls)

        # Use Playwright for fetching
        async with async_playwright() as p:
            browser = await p.chromium.launch(**self.browser_launch_kwargs)
            semaphore = asyncio.Semaphore(self.max_concurrency)

            # Prepare tasks for fetching
            tasks = [
                self._fetch_url(browser, url, semaphore, screenshot)
                for url in urls if force or url not in self.cache
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            await browser.close()

        # Map results to URLs
        url_results = {}
        for url, result in zip(urls, results):
            if isinstance(result, Exception):
                log.error(f"Error fetching {url}: {result}")
            elif result:
                url_results[url] = result
        return url_results

    async def _cleanup_cache(self, urls):
        """
        Clean up caches for given URLs.

        :param urls: List of URLs to clean up.
        """
        for url in urls:
            if url in self.failed_url_cache:
                log.info(f"Removing {url} from failed_url_cache")
                self.failed_url_cache.delete(url)
            if url in self.cache:
                log.info(f"Removing {url} from cache")
                self.cache.delete(url)

    async def _fetch_url(self, browser, url, semaphore, screenshot):
        """Fetch a single URL with retries."""
        async with semaphore:
            context = await browser.new_context(**self.context_kwargs)
            try:
                page = await context.new_page()
                await stealth_async(page)

                # Navigate and handle retries
                if not await navigate_with_retries(page, url, timeout=DEFAULT_TIMEOUT):
                    return url, None

                # Wait for the page to be ready
                await wait_for_page_ready(page, timeout=DEFAULT_TIMEOUT)
                await asyncio.sleep(5)

                html = await page.content()
                if screenshot:
                    await self._save_screenshot(page, url)

                self.cache.set(url, html, expire=self.ttl)
                return url, html

            except Exception as e:
                log.error(f"Error fetching {url}: {e}")
                return url, None
            finally:
                await context.close()

    async def _save_screenshot(self, page, url, postfix=None):
        """Save a screenshot of the page."""
        try:
            filename = f"{url.replace('/', '_').replace(':', '_')}_" \
                       f"{datetime.now().strftime('%Y%m%d_%H%M%S')}{f'_{postfix}' if postfix else ''}.png"
            filepath = os.path.join(self.screenshot_dir, filename)
            await page.screenshot(path=filepath)
            log.info(f"Screenshot saved: {filepath}")
        except Exception as e:
            log.error(f"Failed to save screenshot: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    top_dir = os.path.dirname(os.path.dirname(__file__))
    with open(os.path.join(top_dir, "dataset", "training_urls.json"), "r") as ufp:
        _urls = json.load(ufp)
    fetcher = HTMLFetcher(cache_dir=os.path.join(top_dir, "test_cache"), max_concurrency=os.cpu_count()//2)
    _results = fetcher.fetch_all(_urls, screenshot=True, force=True)
    for _url, _html in _results.items():
        if _html:
            print(f"Successfully fetched {len(_html)} characters from {_url}")
        else:
            print(f"Failed to fetch {_url}")
