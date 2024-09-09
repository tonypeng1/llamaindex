from pathlib import Path
from setuptools import find_packages, setup

# Load packages from requirements.txt
BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt")) as file:
    required_packages = [ln.strip() for ln in file.readlines()]

setup(
    name="llamaindex",
    version=0.1,
    description="RAG using llamaindex",
    author="Tony Peng",
    author_email="tony3t3t@hotmail.com",
    python_requires=">=3.11",
    packages=find_packages(),
    install_requires=required_packages,
)