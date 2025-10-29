from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="fusion-metric",
    version="0.1.0",
    author="HuXiangyu123",
    description="Basic benchmark metrics for image fusion evaluation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HuXiangyu123/Fusion-metric",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
)
