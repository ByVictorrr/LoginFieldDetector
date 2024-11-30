import os
import pytest
from login_field_detector import LoginFieldDetector


@pytest.fixture(scope="session")
def detector():
    """Fixture to initialize and train LoginFieldDetector."""
    detector = LoginFieldDetector()
    detector.train(force=True, screenshots=True)  # Assuming only HTML data is required for training
    return detector


@pytest.mark.parametrize("url", [
    "https://x.com/i/flow/login",  # Twitter
    "https://www.facebook.com/login",  # Facebook
    "https://www.instagram.com/accounts/login/",  # Instagram
    "https://www.linkedin.com/login",  # LinkedIn
    "https://secure.paypal.com/signin",  # PayPal
])
def test_valid_login_urls(detector, url):
    """Test LoginFieldDetector with valid login page URLs."""
    assert detector.predict(url=url), f"Failed to detect login fields for {url}"


@pytest.mark.parametrize("url", [
    "https://example.com/non-login-page",  # Non-login page
    "ftp://example.com",  # Unsupported protocol
    "https://malformed.com",  # Malformed URL
])
def test_invalid_login_urls(detector, url):
    """Test LoginFieldDetector with invalid or non-login URLs."""
    assert not detector.predict(url=url), f"Incorrectly detected login fields for {url}"

