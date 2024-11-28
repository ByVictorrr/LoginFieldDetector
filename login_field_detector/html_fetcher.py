import logging
import time
import random
import requests
from diskcache import Cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import cloudscraper
from fake_useragent import UserAgent

log = logging.getLogger(__name__)


class HTMLFetcher:
    def __init__(self, cache_dir='./html_cache', ttl=7 * 24 * 3600, max_workers=10):
        """HTMLFetcher for downloading HTML content with caching and retry support.

        :param cache_dir: Directory for persistent cache storage.
        :param ttl: Time-to-live (in seconds) for the cache.
        :param max_workers: Number of concurrent threads for fetching.
        """
        self.cache = Cache(cache_dir)  # Persistent cache for successful fetches
        self.failed_url_cache = Cache(f"{cache_dir}/failed_urls")  # Persistent cache for failed URLs
        self.ttl = ttl
        self.max_workers = max_workers
        self.user_agent = UserAgent()

    def create_scraper(self):
        """Create a Cloudscraper instance for handling anti-bot challenges."""
        scraper = cloudscraper.create_scraper(
            browser={"custom": self.user_agent.random}
        )
        scraper.headers.update({
            "Referer": "https://www.google.com",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
        })
        return scraper

    def fetch_html(self, url, force=False, use_scraper=True):
        """Fetch HTML content for a given URL with retry support.

        :param url: URL to fetch.
        :param force: Force fetching even if cached or marked as failed.
        :param use_scraper: Whether to use Cloudscraper for fetching.
        :return: HTML content as a string.
        :raises requests.exceptions.RequestException: If the fetch fails.
        """
        if not force and url in self.failed_url_cache:
            log.warning(f"Skipping previously failed URL: {url}")
            return None

        if not force and url in self.cache:
            log.info(f"Using cached HTML for {url}")
            return self.cache[url]

        log.info(f"Fetching HTML for {url} {'with scraper' if use_scraper else 'with requests'}")
        scraper = self.create_scraper() if use_scraper else requests

        try:
            response = scraper.get(url, timeout=20, allow_redirects=True)
            response.raise_for_status()
            html_content = response.text
            self.cache.set(url, html_content, expire=self.ttl)  # Cache successful fetch
            return html_content
        except requests.exceptions.RequestException as e:
            log.warning(f"Failed to fetch {url}: {e}")
            self.failed_url_cache.set(url, "failed", expire=self.ttl)  # Mark as failed with TTL
            raise

    def fetch_all(self, urls, force=False):
        """Fetch HTML content for multiple URLs concurrently.

        :param urls: List of URLs to fetch.
        :param force: Force fetching even if URLs are cached or marked as failed.
        :return: A dictionary mapping URLs to their HTML content.
        """
        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_url = {
                executor.submit(self.fetch_html, url, force): url
                for url in urls if force or url not in self.failed_url_cache
            }
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    html_content = future.result()
                    if html_content:
                        results[url] = html_content
                except requests.exceptions.RequestException:
                    log.warning(f"Error fetching {url}")
                time.sleep(random.uniform(1, 3))  # Random delay to avoid rate-limiting
        return results
