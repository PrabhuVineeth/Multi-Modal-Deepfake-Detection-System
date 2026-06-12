"""Setup script for the Deepfake Forensic Detection System."""

from setuptools import setup, find_packages

setup(
    name="deepfake-forensics",
    version="1.0.0",
    description="Multimodal Deepfake Forensic Detection System",
    author="Deepfake Forensics Team",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "torchaudio>=2.1.0",
        "transformers>=4.36.0",
        "pytorch-crf>=0.7.2",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "streamlit>=1.29.0",
        "pydantic>=2.5.0",
        "loguru>=0.7.0",
    ],
    entry_points={
        "console_scripts": [
            "deepfake-forensics=demo:main",
        ],
    },
)
