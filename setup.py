from setuptools import setup, find_packages

setup(
    name="mlx_distributed",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Core ML Dependencies
        "mlx>=0.20.0",  
        "numpy>=1.24.0",
        "transformers>=4.36.0",  
        "safetensors>=0.4.0",
        "datasets>=2.15.0",
        "torch>=2.1.0",  
        
        # Distributed Training
        "mpi4py>=3.1.4",
        "paramiko>=3.3.1",
        "pyzmq>=25.1.1",
        "asyncssh>=2.14.0",
        "iperf3>=0.1.11",  
        
        # Storage & IO
        "aiofiles>=23.2.1",
        "aiosqlite>=0.19.0",
        "xxhash>=3.4.1",
        
        # API & Serving
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.5.0",
        
        # Monitoring & Visualization
        "psutil>=5.9.0",
        "plotly>=5.18.0",
        "dash>=2.14.0",
        "dash-core-components>=2.0.0",
        "dash-html-components>=2.0.0",
        "pandas>=2.1.0",
        
        # Utilities
        "tqdm>=4.66.1",         
        "pyyaml>=6.0.1",
        "iperf3>=0.1.11",  # For network testing
        "metal-sdk>=0.5.0",  # For Metal GPU monitoring
        "huggingface-hub>=0.19.4",  
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",  
            "pytest-asyncio>=0.21.1",
            "pytest-cov>=4.1.0",
            "black>=23.11.0", 
            "isort>=5.12.0",
            "mypy>=1.7.0",
            "pylint>=3.0.0",
            # Type Stubs
            "types-PyYAML>=6.0.12",
            "types-tqdm>=4.65.0",
            "types-paramiko>=3.3.1",
            "types-psutil>=5.9.0",
            "types-asyncssh>=2.13.0",
        ],
        "docs": [
            "sphinx>=7.2.0",
            "sphinx-rtd-theme>=1.3.0",
            "sphinx-autodoc-typehints>=1.25.0",
        ],
    },
    python_requires=">=3.12",
    author="Jarrod Barnes",
    author_email="jbarnes850@gmail.com",
    description="Distributed training with MLX for Apple Silicon",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jbarnes850/mlx_distributed_training",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: MacOS :: MacOS X",
    ],
)