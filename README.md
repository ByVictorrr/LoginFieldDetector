# HTML Login Field Detector
[![Coverage Status](https://coveralls.io/repos/github/ByVictorrr/LoginFieldDetector/badge.svg?branch=main)](https://coveralls.io/github/ByVictorrr/LoginFieldDetector?branch=main)

`html-login-field-detector` is a Python library designed to identify and process login fields in HTML documents. Powered by machine learning (DistilBERT) and modern web scraping tools, this library provides a robust solution for automating form detection in web applications.

## Features
- Detects login forms in HTML documents.
- Utilizes Hugging Face's DistilBERT model for advanced text processing.
- Integrates seamlessly with Python web scraping workflows.
- Supports GPU acceleration for faster processing.

## Installation

### Using pip
To install the library along with the CPU-compatible dependencies:
```bash
pip install html-login-field-detector[cpu]
```

For GPU compatibility:
```bash
pip install html-login-field-detector[gpu] --extra-index-url https://download.pytorch.org/whl/cu118
```
### Install System Dependencies
Run the following command to install Playwright's system dependencies:

```bash
playwright install-deps
```


## Usage
```python
from login_field_detector import LoginFieldDetector

# Initialize the detector
detector = LoginFieldDetector()

# Detect login fields in an HTML document
html_source = "<html>...</html>"  # Your HTML content
result = detector.detect(html_source)

print(result)  # Output details of detected login fields
```

## Dataset
This project includes a dataset of login page URLs for training and testing purposes, located at `dataset/training_urls.json`. The dataset can be extended or updated as needed.

## Development
Clone the repository and install the dependencies locally:
```bash
git clone https://github.com/ByVictorrr/LoginFieldDetector.git
cd LoginFieldDetector

# Install dependencies
pip install -e .[gpu,test]
playwright install
```

### Running Tests
Run the tests using `pytest`:
```bash
pytest
```

## License
This project is licensed under the [MIT License](LICENSE).

## Contributing
We welcome contributions! Please fork the repository, make changes, and submit a pull request.

## Links
- **Homepage**: [ByVictorrr on GitHub](https://github.com/ByVictorrr)
- **Repository**: [LoginFieldDetector](https://github.com/ByVictorrr/LoginFieldDetector)
- **Documentation**: [Docs](https://byvictorrr.github.io/LoginFieldDetector)
- **Dataset**: `dataset/training_urls.json`

