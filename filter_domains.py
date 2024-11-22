import json
import asyncio
from urllib.parse import urljoin

import aiohttp
from bs4 import BeautifulSoup
from langdetect import detect
from tqdm.asyncio import tqdm


async def fetch_language(domain, session):
    """Fetch the language of a domain."""
    try:
        async with session.get(f"https://{domain}", timeout=5) as response:
            if response.status == 200:
                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")

                # Check lang attribute in <html>
                html_tag = soup.find("html")
                if html_tag and html_tag.get("lang"):
                    return domain, html_tag.get("lang").split("-")[0]  # Extract base language

                # Fallback: Detect language from text content
                text = soup.get_text(strip=True)
                return domain, detect(text[:1000])
    except Exception:
        return domain, None


async def find_login_url(domain, session, login_keywords):
    """Find login URL for a domain."""
    homepage_url = f"https://{domain.strip()}"
    try:
        async with session.get(homepage_url, timeout=5) as response:
            if response.status == 200:
                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")

                # Find all links
                links = [a["href"] for a in soup.find_all("a", href=True)]

                # Filter for login-related links
                for link in links:
                    if any(keyword in link.lower() for keyword in login_keywords):
                        return urljoin(homepage_url, link)
    except Exception:
        return None


async def main():
    # Load domains
    amount = 10000
    with open("domains_unique.json", "r") as flp:
        domains = json.load(flp)

    # with open("domains_unique.json", "w") as flp:
        # json.dump(domains[amount:], flp, indent=4)


    login_keywords = ["login", "signin", "sign-in", "account/login", "auth/login", "user/login"]

    # Step 1: Filter English domains
    filtered_domains = []
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_language(domain, session) for domain in domains[:amount]]
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Filtering English domains"):
            result = await future  # Await the individual task
            if result is not None:  # Check if result is valid
                domain, lang = result
                if lang == "en":
                    filtered_domains.append(domain)

    # Step 2: Find login URLs
    valid_login_urls = []
    async with aiohttp.ClientSession() as session:
        tasks = [find_login_url(domain, session, login_keywords) for domain in filtered_domains]
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Finding Login URLs"):
            login_url = await future
            if login_url:
                valid_login_urls.append(login_url)

    # Save results to a file
    output_path = "filtered_english_domains.json"
    with open(output_path, "w") as f:
        json.dump(valid_login_urls, f, indent=2)




# Run the event loop
if __name__ == "__main__":
    asyncio.run(main())
