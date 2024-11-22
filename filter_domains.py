import json

import tqdm
import requests
from bs4 import BeautifulSoup
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0  # Ensure reproducibility for langdetect


def fetch_language(domain):
    try:
        response = requests.get(f"https://{domain}", timeout=5)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            # Check the lang attribute in the <html> tag
            html_tag = soup.find('html')
            if html_tag and html_tag.get('lang'):
                return html_tag.get('lang').split('-')[0]  # Extract base language (e.g., 'en' from 'en-US')

            # Fallback: Detect language from text content
            text = soup.get_text(strip=True)
            return detect(text[:1000])  # Analyze first 1000 characters for efficiency
        else:
            return None
    except Exception as e:
        return None  # Handle errors (e.g., timeout, connection errors)


# Filtered list to further process
filtered_domains = []

with open("domains_unique.json", "r") as flp:
    domains = json.load(flp)

# with open("domains_unique.json", "w") as flp:
    # json.dump(domains[1500:], flp, indent=4)

for domain in tqdm.tqdm(domains[:1500]):
    lang = fetch_language(domain)
    if lang == 'en':  # Keep only domains with English content
        filtered_domains.append(domain)

# Keywords for login
login_keywords = ["login", "signin", "sign-in", "account/login", "auth/login", "user/login"]

# Store results
valid_login_urls = []

# Check URLs for each domain
for domain in tqdm.tqdm(filtered_domains, desc="Finding Login URLs"):
    homepage_url = f"https://{domain.strip()}"
    try:
        response = requests.get(homepage_url, timeout=5)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find all links
            links = [a['href'] for a in soup.find_all('a', href=True)]

            # Filter for login-related links
            for link in links:
                if any(keyword in link.lower() for keyword in login_keywords):
                    # Handle relative URLs
                    if link.startswith('/'):
                        link = homepage_url + link
                    valid_login_urls.append(link)
                    break  # Found a valid login URL, move to the next domain
    except requests.RequestException:
        continue

# Save results to a file
output_path = "filtered_english_domains.json"
with open(output_path, 'w') as f:
    json.dump(valid_login_urls, f, indent=2)



