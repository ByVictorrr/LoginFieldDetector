import logging
import asyncio
from datetime import datetime
from os import path, makedirs, cpu_count

from diskcache import Cache
from fake_useragent import UserAgent
from undetected_playwright import stealth_async
from undetected_playwright_patch import patch_sync_playwright

log = logging.getLogger(__name__)


def playwright_timeout(seconds):
    """Convert a given time in seconds to milliseconds."""
    return seconds * 1000


async def is_cloudflare_challenge_page(page):
    """
    Detects if the current page is a Cloudflare challenge page.
    """
    try:
        footer_text = await page.query_selector("#footer-text")
        content = await page.content()
        if footer_text:
            footer_content = await footer_text.inner_text()
            if "performance & security by cloudflare" in footer_content.lower():
                log.info("Cloudflare footer detected.")
                if "verify you are human" in content.lower():
                    checkbox = await page.query_selector('input[type="checkbox"]')
                    verifying_msg = await page.query_selector('#verifying-msg')
                    iframe = await page.query_selector('iframe[src*="/cdn-cgi/challenge-platform/"]')
                    if checkbox or verifying_msg or iframe:
                        log.info("Cloudflare challenge page confirmed.")
                        return True
        return False
    except Exception as e:
        log.warning(f"Error detecting Cloudflare challenge page: {e}")
        return False


async def handle_cloudflare_challenge(page):
    """Attempts to solve Cloudflare challenges."""
    try:
        challenge_checkbox = await page.query_selector('input[type="checkbox"]')
        if challenge_checkbox:
            log.info("Cloudflare challenge detected. Clicking checkbox...")
            await challenge_checkbox.click()
            await page.wait_for_selector("#verifying", state="hidden", timeout=playwright_timeout(10))
            log.info("Verification spinner disappeared.")
            await page.wait_for_selector("#success", timeout=playwright_timeout(10))
            log.info("Challenge solved successfully.")
            return True
        return False
    except Exception as e:
        log.error(f"Error handling Cloudflare challenge: {e}")
        return False


class HTMLFetcher:
    def __init__(
        self,
        cache_dir=None,
        ttl=7 * 24 * 3600,
        max_concurrency=cpu_count(),
        browser_launch_kwargs=None,
        context_kwargs=None,
    ):
        cache_dir = cache_dir or path.join(path.dirname(__file__), "html_cache")
        makedirs(cache_dir, exist_ok=True)

        self.cache = Cache(cache_dir)
        self.failed_url_cache = Cache(f"{cache_dir}/failed_urls")
        self.screenshot_dir = path.join(cache_dir, "screenshots")
        makedirs(self.screenshot_dir, exist_ok=True)

        self.ttl = ttl
        self.max_concurrency = max_concurrency

        self.browser_launch_kwargs = browser_launch_kwargs or {
            "headless": False,
            "args": ["--disable-http2", "--disable-blink-features=AutomationControlled"],
        }

        self.context_kwargs = context_kwargs or {
            "user_agent": UserAgent().random,
            "viewport": {"width": 1920, "height": 1080},
            "ignore_https_errors": True,
        }

    async def fetch_all(self, urls, force=False, screenshot=False):
        """Fetch multiple URLs concurrently."""
        if force:
            for url in urls:
                self.cache.delete(url)
                self.failed_url_cache.delete(url)

        with patch_sync_playwright() as p:
            browser = await p.chromium.launch(**self.browser_launch_kwargs)
            semaphore = asyncio.Semaphore(self.max_concurrency)

            tasks = [
                self._fetch_url_with_context(browser, url, semaphore, screenshot)
                for url in urls
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
                log.info(f"Fetching: {url}")

                if not await self._navigate_with_retries(page, url):
                    return url, None

                await self._wait_for_page_ready(page)
                if await is_cloudflare_challenge_page(page):
                    if not await handle_cloudflare_challenge(page):
                        return url, None

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

    async def _navigate_with_retries(self, page, url, retries=3):
        """Navigate to URL with retries."""
        for attempt in range(retries):
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                return True
            except Exception as e:
                log.warning(f"Retry {attempt + 1} for {url}: {e}")
                await asyncio.sleep(2 ** attempt)
        return False

    async def _wait_for_page_ready(self, page):
        """Wait for the page to stabilize."""
        try:
            await page.wait_for_load_state("networkidle")
            await page.wait_for_function("document.readyState === 'complete'")
            log.info("Page fully loaded.")
        except Exception as e:
            log.warning(f"Error waiting for page readiness: {e}")

    async def _save_screenshot(self, page, url):
        """Save a screenshot."""
        filename = f"{url.replace('/', '_').replace(':', '_')}.png"
        filepath = path.join(self.screenshot_dir, filename)
        await page.screenshot(path=filepath)
        log.info(f"Screenshot saved to {filepath}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fetcher = HTMLFetcher()
    asyncio.run(fetcher.fetch_all(["https://example.com"], screenshot=True))

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
