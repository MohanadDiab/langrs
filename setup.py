from setuptools import setup, find_packages

setup(
    name="langrs_segmentation",
    version="0.1.0",
    author="Your Name",
    author_email="your_email@example.com",
    description="Remote sensing image segmentation using LangSAM",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/langrs_segmentation",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "Pillow",
        "rasterio",
        "torch",
        "scipy",
        "sklearn",
        "samgeo",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
