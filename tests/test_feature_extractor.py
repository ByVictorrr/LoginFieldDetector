# test_html_feature_extractor.py
import pytest
from bs4 import BeautifulSoup
from login_field_detector import determine_label


@pytest.mark.parametrize(
    "html_snippet, expected_label",
    [
        # Test cases for valid labels
        ('<input type="text" name="username">', "USERNAME"),
        ('<input type="password" name="pass">', "PASSWORD"),
        ('<button>Login</button>', "LOGIN_BUTTON"),
        ('<input type="text" name="phone">', "PHONE_NUMBER"),
        ('<input type="text" placeholder="Enter captcha">', "CAPTCHA"),
        ('<a href="#">Forgot password?</a>', "FORGOT_PASSWORD"),
        ('<button>Sign in with Google</button>', "SOCIAL_LOGIN_BUTTONS"),
        ('<div>I\'m not a robot</div>', "CAPTCHA"),

        # Test cases for default label (UNLABELED)
        ('<div>Completely unrelated text</div>', "UNLABELED"),
        ('<span style="display: none;">Hidden content</span>', "UNLABELED"),
    ]
)
def test_determine_label(html_snippet, expected_label):
    """Test determine_label function with various HTML inputs."""
    soup = BeautifulSoup(html_snippet, "lxml")
    tag = soup.find()
    assert determine_label(tag) == expected_label


def test_hidden_input_is_unlabeled():
    """Ensure hidden inputs are labeled as UNLABELED."""
    html_snippet = '<input type="hidden" name="token" value="abcd1234">'
    soup = BeautifulSoup(html_snippet, "lxml")
    tag = soup.find()
    assert determine_label(tag) == "UNLABELED"

