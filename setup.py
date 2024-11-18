from setuptools import setup, find_packages

setup(
    name="mlx_distributed",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Core ML Dependencies
        "mlx>=0.20.0",
        "numpy>=1.24.0",
        "torch>=2.1.0",  # For data preprocessing
        
        # Distributed Training
        "mpi4py>=3.1.4",
        "paramiko>=3.3.1",  # For SSH operations
        
        # Data Processing
        "datasets>=2.15.0",
        "transformers>=4.35.0",
        "huggingface-hub>=0.19.0",
        "safetensors>=0.4.0",
        
        # API and Serving
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.5.0",
        
        # Monitoring & Visualization
        "psutil>=5.9.0",
        "plotly>=5.18.0",
        "dash>=2.14.0",
        "pandas>=2.1.0",  # Required for dashboard
        "dash-core-components>=2.0.0",
        "dash-html-components>=2.0.0",
        
        # Utilities
        "tqdm>=4.65.0",
        "pyyaml>=6.0.1",
        "iperf3>=0.1.11",  # For network testing
        "aiofiles>=23.2.1",  # For async file operations
        "metal-sdk>=0.5.0",  # For Metal GPU monitoring
    ],
    extras_require={
        "dev": [
            # Testing
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.1",
            "pytest-cov>=4.1.0",
            
            # Code Quality
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.7.0",
            "types-PyYAML>=6.0.12",
            "types-tqdm>=4.65.0",
            "types-paramiko>=3.3.1",
            "types-aiofiles>=23.2.0",
            "types-psutil>=5.9.0",
            
            # Documentation
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
        ]
    },
    python_requires=">=3.10",
    author="Jarrod Barnes",
    author_email="jbarnes850@gmail.com",
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