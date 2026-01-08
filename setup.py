"""
Setup script for Bahlib - Biometric Anonymizer Hub Library.
Provides backwards compatibility with older pip/setuptools versions.
"""

from setuptools import setup, find_packages

setup(
    name="bahlib",
    version="0.1.0",
    description="Biometric Anonymizer Hub Library - 100% local face anonymization",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Bahlib Contributors",
    license="MIT",
    python_requires=">=3.8",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=[
        "opencv-python>=4.5.0",
        "mediapipe>=0.10.0",
        "numpy>=1.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "bahlib=bahlib.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Security",
    ],
)
