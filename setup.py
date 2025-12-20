"""
Setup script for WBC-Bench-2026 package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = requirements_file.read_text().strip().split("\n")
    requirements = [r.strip() for r in requirements if r.strip() and not r.startswith("#")]

setup(
    name="wbc-bench-2026",
    version="1.0.0",
    author="Senior Researcher",
    description="White Blood Cell Classification for Kaggle WBC-Bench-2026 Competition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ansulx/White-Blood-Cell-Classification",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "wbc-train=scripts.train:main",
            "wbc-inference=scripts.inference:main",
            "wbc-explore=scripts.explore_data:analyze_dataset",
        ],
    },
)

