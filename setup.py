from setuptools import setup, find_packages

setup(
    name="transformer-kernels",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.3.0",
        "triton>=2.3.0",
        "numpy>=1.26.0",
        "einops>=0.8.0",
    ],
    python_requires=">=3.10",
)