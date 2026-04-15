"""Setup configuration for LangRS package."""

from pathlib import Path

from setuptools import find_packages, setup


def _read_requirements(filename: str) -> list[str]:
    path = Path(__file__).parent / filename
    if not path.exists():
        return []
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

install_requires = _read_requirements("requirements-core.txt")

extras_require = {
    # Grounding DINO (optional; may conflict with Rex-Omni / torch stack — use a separate venv if needed).
    "dino": _read_requirements("requirements-dino.txt"),
    # Optional heavy GPU extras for Rex-Omni: flash-attn / vLLM / Triton. The Rex-Omni
    # wrapper code itself is vendored inside langrs; this extra is only for deps.
    "rex-omni": [
        "flash-attn>=2.7.4",
        "vllm>=0.9.1; sys_platform != 'win32'",
        "triton; sys_platform != 'win32'",
        "triton-windows; sys_platform == 'win32'",
    ],
    # Optional vLLM-only extra (without flash-attn), if users just want that backend.
    "vllm": [
        "vllm>=0.9.1; sys_platform != 'win32'",
        "triton; sys_platform != 'win32'",
        "triton-windows; sys_platform == 'win32'",
    ],
}

setup(
    name="langrs",
    version="2.0.0",
    author="Mohanad Diab",
    author_email="mohanad.y.diab@gmail.com",
    description=(
        "Extensible Python package for zero-shot segmentation of aerial imagery "
        "(Rex-Omni or Grounding DINO + SAM)"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MohanadDiab/langrs",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    install_requires=install_requires,
    extras_require=extras_require,
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
    keywords=(
        "remote sensing, computer vision, segmentation, SAM, Rex-Omni, "
        "GroundingDINO, geospatial, AI"
    ),
    project_urls={
        "Bug Reports": "https://github.com/MohanadDiab/langrs/issues",
        "Source": "https://github.com/MohanadDiab/langrs",
        "Documentation": "https://github.com/MohanadDiab/langrs",
    },
)
