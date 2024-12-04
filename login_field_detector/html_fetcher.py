import logging
import asyncio
from datetime import datetime
from os import path, makedirs, cpu_count

from diskcache import Cache
from fake_useragent import UserAgent
from playwright.async_api import async_playwright, BrowserContext, Page
from playwright_stealth import stealth_async

log = logging.getLogger(__name__)


def playwright_timeout(seconds):
    """Convert a given time in seconds to milliseconds.

    :param seconds: The number of seconds.
    :return: The equivalent time in milliseconds.
    """
    return seconds * 1000


async def is_cloudflare_challenge_page(page):
    """
    Checks if the current page is a Cloudflare challenge page.

    :param page: Playwright page instance.
    :return: True if it's a Cloudflare challenge page, False otherwise.
    """
    try:
        # Step 1: Check for the Cloudflare footer
        footer_text = await page.query_selector("#footer-text")
        if footer_text:
            footer_content = await footer_text.inner_text()
            if "performance & security by cloudflare" in footer_content.lower():
                log.info("Cloudflare footer detected.")

                # Step 2: Check for "Verify you are human" text
                content = await page.content()
                if "verify you are human" in content.lower():
                    log.info("Cloudflare 'Verify you are human' text detected.")

                    # Step 3: Look for specific elements
                    checkbox = await page.query_selector('input[type="checkbox"]')
                    verifying_msg = await page.query_selector('#verifying-msg')
                    iframe = await page.query_selector('iframe[src*="/cdn-cgi/challenge-platform/"]')

                    # If any Cloudflare-specific element is found, confirm challenge page
                    if checkbox or verifying_msg or iframe:
                        log.info("Cloudflare challenge page confirmed via specific elements.")
                        return True

        # If no indicators match, assume it's not a Cloudflare challenge page
        log.info("No Cloudflare challenge detected.")
        return False
    except Exception as e:
        log.warning(f"Error detecting Cloudflare challenge page: {e}")
        return False


async def handle_cloudflare_challenge(page):
    """Detects and solves the Cloudflare challenge page.

    :param page: Playwright page instance.
    :return: True if challenge solved successfully, False otherwise.
    """
    try:
        # Check for the challenge checkbox
        challenge_checkbox = await page.query_selector('input[type="checkbox"]')
        if challenge_checkbox:
            log.info("Cloudflare challenge detected. Attempting to solve...")

            # Click the checkbox
            await challenge_checkbox.click()

            # Wait for the "verifying" spinner to disappear
            try:
                await page.wait_for_selector("#verifying", state="hidden", timeout=playwright_timeout(5))
                log.info("Verifying spinner disappeared.")
            except Exception as e:
                log.warning(f"Spinner did not disappear in time: {e}")

            # Wait for the "success" state
            await page.wait_for_selector("#success", timeout=playwright_timeout(5))
            log.info("Cloudflare challenge solved successfully.")
            return True

        log.info("No Cloudflare challenge detected.")
        return False

    except Exception as e:
        log.error(f"Error handling Cloudflare challenge: {e}")
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
        cache_dir = cache_dir or path.join(path.dirname(__file__), "html_cache")
        makedirs(cache_dir, exist_ok=True)

        self.cache = Cache(cache_dir)
        self.failed_url_cache = Cache(f"{cache_dir}/failed_urls")
        self.screenshot_dir = path.join(cache_dir, "screenshots")
        makedirs(self.screenshot_dir, exist_ok=True)

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
            "viewport": {"width": 1920, "height": 1080},
            "ignore_https_errors": True,
            "extra_http_headers": {"accept-language": "en-US,en;q=0.9"},
        }

    async def fetch_all(self, urls, force=False, screenshot=False):
        # Fetch logic remains unchanged
        if force:
            for url in urls:
                # Remove the URL from the failed URL cache if it exists
                if url in self.failed_url_cache:
                    log.info(f"Deleting {url} from failed_url_cache")
                    self.failed_url_cache.delete(url)

                # Remove the URL from the main cache if it exists
                if url in self.cache:
                    log.info(f"Deleting {url} from cache")
                    self.cache.delete(url)

        async with async_playwright() as p:
            browser = await p.chromium.launch(**self.browser_launch_kwargs)
            semaphore = asyncio.Semaphore(self.max_concurrency)

            tasks = [
                self._fetch_url_with_context(browser, url, semaphore, screenshot)
                for url in urls if force or url not in self.cache
            ]
            results = await asyncio.gather(*tasks)
            await browser.close()
            return {url: html for url, html in results if html}

    async def _fetch_url_with_context(self, browser, url, semaphore, screenshot):
        async with semaphore:
            context = await browser.new_context(**self.context_kwargs)
            try:
                page = await context.new_page()
                await stealth_async(page)

                # Navigate to the URL
                if not await self._navigate_with_retries(page, url):
                    return url, None

                # Wait for the page to fully load
                await self._wait_for_page_ready(page, timeout=playwright_timeout(10))

                # Check if it's a Cloudflare challenge page
                if await is_cloudflare_challenge_page(page):
                    log.info(f"Cloudflare challenge page detected for {url}.")
                    # Optionally handle the challenge
                    if not await handle_cloudflare_challenge(page):
                        log.error(f"Failed to solve Cloudflare challenge for {url}.")
                        return url, None
                else:
                    log.info(f"No Cloudflare challenge detected for {url}.")

                # Wait for the page to fully load
                await self._wait_for_page_ready(page, timeout=playwright_timeout(10))

                # Get the page content
                html = await page.content()

                # Save a screenshot if requested
                if screenshot:
                    await self._save_screenshot(page, url, postfix="success")

                # Cache the content
                self.cache.set(url, html, expire=self.ttl)
                return url, html

            except Exception as e:
                log.error(f"Error fetching {url}: {e}")
                return url, None

            finally:
                await context.close()

    async def _navigate_with_retries(self, page, url, retries=3, backoff=2):
        """Navigate to a URL with retries and exponential backoff."""
        for attempt in range(retries):
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                return True
            except Exception as e:
                log.warning(f"Retry {attempt + 1} for {url} failed: {e}")
                await asyncio.sleep(backoff ** attempt)
        return False

    async def _wait_for_page_ready(self, page, timeout=60000):
        """Wait for the page to be fully loaded using multiple checks."""
        try:
            # Step 1: Wait for the network to be idle
            await page.wait_for_load_state("networkidle", timeout=timeout)
            log.info("Network is idle.")

            # Step 2: Wait for the DOM to be fully loaded
            await page.wait_for_function("document.readyState === 'complete'", timeout=timeout)
            log.info("DOM is ready.")

            # Step 3: Wait for spinners or loading indicators to disappear
            spinner_selectors = [".spinner", ".loading", "[aria-busy='true']"]
            for selector in spinner_selectors:
                try:
                    await page.wait_for_selector(selector, state="detached", timeout=timeout)
                    log.info(f"Spinner {selector} disappeared.")
                except asyncio.TimeoutError:
                    log.warning(f"Spinner {selector} did not disappear.")

            # Step 4: Wait for dynamic content to stabilize
            if not await self._wait_for_dynamic_content(page):
                log.warning("Dynamic content did not stabilize.")
        except Exception as e:
            log.error(f"Page readiness failed: {e}")

    async def _wait_for_dynamic_content(self, page, max_retries=10, interval=1000):
        """
        Waits for dynamic content to stop updating.

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

    async def _save_screenshot(self, page, url, postfix=None):
        """Save a screenshot of the page."""
        try:
            filename = f"{url.replace('/', '_').replace(':', '_')}_" \
                       f"{datetime.now().strftime('%Y%m%d_%H%M%S')}{f'_{postfix}' if postfix else ''}.png"
            filepath = path.join(self.screenshot_dir, filename)
            await page.screenshot(path=filepath)
            log.info(f"Screenshot saved: {filepath}")
        except Exception as e:
            log.error(f"Failed to save screenshot: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    fetcher = HTMLFetcher(
        browser_launch_kwargs={
            "headless": False,
            "args": ["--disable-http2", "--start-maximized"],
        },
        context_kwargs={
            "viewport": {"width": 1280, "height": 720},
            "ignore_https_errors": True,
        },
    )

    _urls = ["https://example.com"]
    _results = asyncio.run(fetcher.fetch_all(_urls, force=True, screenshot=True))

    for _url, _html in _results.items():
        if _html:
            print(f"Successfully fetched {len(_html)} characters from {_url}")
        else:
            print(f"Failed to fetch {_url}")
