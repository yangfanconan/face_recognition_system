"""
DDFD-FaceRec - Dual-Domain Feature Decoupling Face Recognition System
端到端人脸识别全系统
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ddfd-face-rec",
    version="0.1.0",
    author="FaceRec Team",
    author_email="facerec@example.com",
    description="端到端人脸识别全系统 - 双分支特征解耦方案",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/ddfd-face-rec",
    packages=find_packages(exclude=["tests", "tools", "deployment"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "face-detect=tools.train_detection:main",
            "face-recognize=tools.train_recognition:main",
            "face-finetune=tools.finetune:main",
            "face-export=deployment.export_onnx:main",
            "face-api=api.main:main",
        ],
    },
)
