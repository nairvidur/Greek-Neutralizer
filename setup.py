"""Setup configuration for greeks-neutralizer package."""

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="greeks-neutralizer",
    version="1.0.0",
    author="Greeks Neutralizer Project",
    description="Multi-position Delta-Gamma-Vega neutralization engine for European options",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/greeks-neutralizer",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[],  # Pure Python, no dependencies
    extras_require={
        "dev": ["pytest>=7.0", "python-docx>=1.0"],
        "report": ["python-docx>=1.0"],
    },
    entry_points={
        "console_scripts": [
            "greeks-neutralizer=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="options greeks delta gamma vega hedging black-scholes neutralization",
)
