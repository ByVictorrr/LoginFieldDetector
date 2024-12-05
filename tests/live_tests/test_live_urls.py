import pytest
from login_field_detector import LoginFieldDetector

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
