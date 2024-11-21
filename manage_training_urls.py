import json
import os

import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

# Common login paths to test
login_paths = ["/login", "/signin", "/account/login", "/auth", "/user/login"]


# Function to check for login page
def find_login_page(domain):
    login_url = None
    for path in login_paths:
        url = f"https://{domain}{path}"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                # Check if the page contains login-related form elements
                soup = BeautifulSoup(response.content, "html.parser")
                if soup.find("form") and soup.find("input", {"type": "password"}):
                    login_url = url
                    break
        except requests.RequestException:
            continue
    return domain, login_url


import requests
from langdetect import detect

def is_english_website(domain):
    try:
        response = requests.get(f"https://{domain}", timeout=5)
        if response.status_code == 200:
            lang = detect(response.text)
            return lang is not None and lang == "en"  # "en" stands for English
    except Exception as e:
        pass
    return False

# Use multithreading for faster processing
def process_domains(domains):
    results = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        for domain, login_url in executor.map(find_login_page, domains):
            if login_url:
                results[domain] = login_url
    return results


# Process domains and print results
if __name__ == "__main__":
    # List of domains to process
    login_pages = dict()
    with open("domains_unique.json", "r") as flp:
        domains = json.load(flp)
        # Use ThreadPoolExecutor to process domains in parallel
    english_domains = []
    for d in tqdm(domains[:1000]):
        if is_english_website(d):
            english_domains.append(d)
    with open("english_domains_unique.json", "w") as flp:
        json.dump(english_domains, flp, indent=4)
    for domain in domains:
        login_pages[domain] = find_login_page(domain)
    for domain, url in login_pages.items():
        print(f"Login page found for {domain}: {url}")
