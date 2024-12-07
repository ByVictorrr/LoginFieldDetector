import os

import pytest
from login_field_detector import LoginFieldDetector

LOGIN_PAGE_ELEMENTS = [
    "USERNAME",
    "PHONE_NUMBER",
    "PASSWORD",
    "LOGIN_BUTTON",
    "TWO_FACTOR_AUTH",
    "SOCIAL_LOGIN_BUTTONS",
]

LIVE_LOGIN_URLS = [
    "https://x.com/i/flow/login",
    "https://www.facebook.com/login",
    "https://www.instagram.com/accounts/login/",
    "https://www.linkedin.com/login",
    "https://secure.paypal.com/signin",
]

NON_LOGIN_URLS = [
    "https://example.com",
    "https://www.wikipedia.org",
    "https://news.ycombinator.com",
]


@pytest.mark.external
class TestDetectorExternal:
    @pytest.fixture(scope="class")
    def detector(self):
        """Fixture to initialize the LoginFieldDetector."""
        detector = LoginFieldDetector()
        detector.train(force=True, screenshots=True)
        return detector

    @pytest.mark.parametrize("url", [
        "https://x.com/i/flow/login",  # Twitter
        "https://www.facebook.com/login",  # Facebook
        "https://www.instagram.com/accounts/login/",  # Instagram
        "https://www.linkedin.com/login",  # LinkedIn
        "https://secure.paypal.com/signin",  # PayPal
    ])
    def test_valid_login_urls(self, detector, url):
        """Test valid login pages."""
        values = detector.predict(url=url)
        assert any(
            len(v) > 0 for k, v in values.items() if k in LOGIN_PAGE_ELEMENTS
        ), f"Failed to find login fields on {url}"

    @pytest.mark.parametrize("url", [
        "https://example.com/non-login-page",
        "ftp://example.com",
        "https://malformed.com",
    ])
    def test_invalid_login_urls(self, detector, url):
        """Test non-login or invalid URLs."""
        values = detector.predict(url=url)
        assert not any(values.values()), f"Incorrectly detected login fields for {url}"


class TestDetectorInternal:
    @pytest.fixture(scope="class")
    def detector(self):
        """Fixture to initialize the LoginFieldDetector."""
        detector = LoginFieldDetector()
        # detector.train(force=False, screenshots=False)
        return detector

    @pytest.mark.parametrize("html_file", [
        "crunchyroll.html",
        "dailymotion.html",
        "etsy.html",
    ])
    def test_valid_html_detection(self, detector, html_file):
        """Test detector with valid cached HTML files."""
        file_path = os.path.join(os.path.dirname(__file__), "feature_extraction", "valid", html_file)
        assert os.path.exists(file_path), f"{html_file} does not exist!"

        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        values = detector.predict(html_content=html_content)
        assert any(
            len(v) > 0 for k, v in values.items() if k in LOGIN_PAGE_ELEMENTS


        ), f"Failed to detect login fields in {html_file}"

    @pytest.mark.parametrize("html_file", [
        "malformed.html",
        "non_login.html",
        "uncommon_attributes.html",
    ])
    def test_invalid_html_detection(self, detector, html_file):
        """Test detector with invalid cached HTML files."""
        file_path = os.path.join(os.path.dirname(__file__), "feature_extraction", "invalid", html_file)
        assert os.path.exists(file_path), f"{html_file} does not exist!"

        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        values = detector.predict(html_content=html_content)
        assert not any(
            len(v) > 0 for result in values.items() for k, v in result.items() if k in LOGIN_PAGE_ELEMENTS
        ), f"Incorrectly detected login fields in {html_file}"


@pytest.mark.external
@pytest.mark.parametrize("url", LIVE_LOGIN_URLS)
def test_live_login_pages(url):
    """Test LoginFieldDetector on live login pages."""
    detector = LoginFieldDetector()
    values = detector.predict(url=url)
    assert any(len(v) > 0 for v in values.values()), f"No login fields detected for {url}"


@pytest.mark.external
@pytest.mark.parametrize("url", NON_LOGIN_URLS)
def test_non_login_pages(url):
    """Ensure LoginFieldDetector does not falsely detect login fields."""
    detector = LoginFieldDetector()
    values = detector.predict(url=url)
    assert not any(v for v in values.values()), f"Incorrectly detected login fields for {url}"




