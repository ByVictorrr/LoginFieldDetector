import pytest
from login_field_detector import LoginFieldDetector

LGOIN_PAGE_ELEMENTS = [
    "USERNAME",
    "PHONE_NUMBER",
    "PASSWORD",
    "LOGIN_BUTTON",
    "TWO_FACTOR_AUTH",
    "SOCIAL_LOGIN_BUTTONS",
]


class TestDetectorProjections(pytest.Class):
    def setup(self) -> None:
        self.detector = LoginFieldDetector()
        self.detector.train(force=False, screenshots=True)  # Assuming only HTML data is required for training

    @pytest.mark.parametrize("url", [
        "https://x.com/i/flow/login",  # Twitter
        "https://www.facebook.com/login",  # Facebook
        "https://www.instagram.com/accounts/login/",  # Instagram
        "https://www.linkedin.com/login",  # LinkedIn
        "https://secure.paypal.com/signin",  # PayPal
    ])
    def test_valid_login_urls(self, url):
        values = self.detector.predict(url=url)
        if not any(len(v) > 0 for k, v in values.items() if k in LoginFieldDetector):
            pytest.fail(f"The detector could not find any login fields on {url=}")

    @pytest.mark.parametrize("url", [
        "https://example.com/non-login-page",  # Non-login page
        "ftp://example.com",  # Unsupported protocol
        "https://malformed.com",  # Malformed URL
    ])
    def test_invalid_login_urls(self, url):
        """Test LoginFieldDetector with invalid or non-login URLs."""
        assert not self.detector.predict(url=url), f"Incorrectly detected login fields for {url}"
