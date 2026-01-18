"""Setup configuration for LangRS package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    requirements = []

setup(
    name="langrs",
    version="2.0.0",  # Major version bump for refactored architecture
    author="Mohanad Diab",
    author_email="mdiab@eurac.edu",
    description="A modern, extensible Python package for zero-shot segmentation of aerial images using GroundingDINO and SAM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MohanadDiab/langrs",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    install_requires=requirements,
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="remote sensing, computer vision, segmentation, SAM, GroundingDINO, geospatial, AI",
    project_urls={
        "Bug Reports": "https://github.com/MohanadDiab/langrs/issues",
        "Source": "https://github.com/MohanadDiab/langrs",
        "Documentation": "https://github.com/MohanadDiab/langrs",
    },
)
