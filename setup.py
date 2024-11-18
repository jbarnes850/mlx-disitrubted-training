from setuptools import setup, find_packages

setup(
    name="mlx_distributed",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "mlx>=0.20.0",
        "mpi4py>=3.1.4",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "datasets>=2.15.0",
        "transformers>=4.35.0",
        "huggingface-hub>=0.19.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.5.0",
        "psutil>=5.9.0",
        "safetensors>=0.4.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "mpi4py-mpich>=3.1.4",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.7.0",
        ]
    },
    python_requires=">=3.10",
    author="Jarrod Barnes",
    author_email="jbarnes850",
    description="Distributed training with MLX for Apple Silicon",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jbarnes850/mlx_distributed",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

# requirements.txt
#mlx>=0.20.0
#numpy>=1.24.0
#tqdm>=4.65.0
#pytest>=7.0.0