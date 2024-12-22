from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="langrs",
    version="0.1.0",
    author="Mohanad Diab",
    author_email="mohanad.y.diab@gmail.com",
    description="A language-driven remote sensing image analysis package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mohanaddiab/langrs",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "matplotlib",
        "pillow",
        "rasterio",
        "scikit-learn",
        "torch",
        "samgeo",
        "seaborn",
    ],
)