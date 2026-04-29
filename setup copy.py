"""
setup.py — LLM-FMEA Supply Chain Risk Assessment Framework
"""
from setuptools import setup, find_packages

setup(
    name="llm-fmea-scra",
    version="1.0.0",
    author="Nikhil Reddy Donapati",
    author_email="nikhil.donapati@myemail.indwes.edu",
    description=(
        "LLM-Augmented Decision Intelligence for Real-Time Supply Chain "
        "Risk Assessment: A Parameterized Simulation and FMEA-Based Framework"
    ),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/nikhildonapati/llm-fmea-scra",
    packages=find_packages(exclude=["tests*", "experiments*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.4.0",
        "openai>=1.3.0",
        "tqdm>=4.66.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-asyncio", "black", "flake8", "mypy"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Scheduling",
    ],
    keywords=[
        "supply-chain", "risk-assessment", "fmea", "llm",
        "decision-intelligence", "enterprise-ai", "fastapi"
    ],
    entry_points={
        "console_scripts": [
            "scra-api=api.main:app",
            "scra-simulate=experiments.run_simulation:run_simulation_experiment",
        ]
    },
)
