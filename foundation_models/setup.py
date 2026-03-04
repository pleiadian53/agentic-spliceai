"""
Setup script for foundation_models package.

Alternative to pyproject.toml for backwards compatibility.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")
else:
    long_description = "Foundation model integration for agentic-spliceai"

setup(
    name="agentic-spliceai-foundation-models",
    version="0.1.0",
    description="Foundation model integration for agentic-spliceai (Evo2, Evo1, Nucleotide Transformer)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="barnettchiu@gmail.com",
    url="https://github.com/pleiadian53/agentic-spliceai",
    packages=find_packages(),
    python_requires=">=3.12",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "einops>=0.6.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "h5py>=3.8.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "quantization": [
            "bitsandbytes>=0.41.0",
            "accelerate>=0.20.0",
        ],
        "adapters": [
            "peft>=0.4.0",
        ],
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.23.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.12",
    ],
)
