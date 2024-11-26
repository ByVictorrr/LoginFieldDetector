from concurrent.futures import ThreadPoolExecutor, as_completed
import cloudscraper
import requests
from fake_useragent import UserAgent
import redis
import hashlib
from tenacity import retry, wait_exponential, stop_after_attempt


class RedisDatasetCache:
    def __init__(self, redis_host="localhost", redis_port=6379, redis_db=0, ttl_seconds=24 * 3600):
        self.redis = redis.StrictRedis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True)
        self.ttl_seconds = ttl_seconds

    def get(self, url):
        cache_key = self._generate_cache_key(url)
        return self.redis.get(cache_key)

    def set(self, url, content):
        cache_key = self._generate_cache_key(url)
        self.redis.setex(cache_key, self.ttl_seconds, content)

    def _generate_cache_key(self, url):
        return hashlib.md5(url.encode()).hexdigest()


class HTMLFetcher:
    def __init__(self, cache=None, max_workers=10):
        self.cache = cache or RedisDatasetCache()
        self.max_workers = max_workers
        self.user_agent = UserAgent()

    def create_scraper(self):
        return cloudscraper.create_scraper(
            browser={"custom": self.user_agent.random},
            headers={
                "Referer": "https://www.google.com",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
            },
        )

    @retry(wait=wait_exponential(multiplier=1, min=4, max=30), stop=stop_after_attempt(5))
    def fetch_html(self, url):
        cached_content = self.cache.get(url)
        if cached_content:
            print(f"Using cached HTML for {url}")
            return cached_content

        print(f"Fetching HTML for {url}")
        scraper = self.create_scraper()
        try:
            response = scraper.get(url, timeout=20, allow_redirects=True)
            response.raise_for_status()
            html = response.text
            self.cache.set(url, html)
            return html
        except requests.exceptions.SSLError as e:
            print(f"SSL error for {url}: {e}")
        except requests.exceptions.TooManyRedirects as e:
            print(f"Too many redirects for {url}: {e}")
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            raise

    def fetch_all(self, urls):
        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_url = {executor.submit(self.fetch_html, url): url for url in urls}
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    if html := future.result():
                        results[url] = html
                except Exception as e:
                    print(f"Error processing {url}: {e}")
        return results
