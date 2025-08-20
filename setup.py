from pathlib import Path
from setuptools import setup

# Read dependencies from requirements.txt to keep a single source of truth
req_path = Path(__file__).parent / "requirements.txt"
install_requires = []
if req_path.exists():
    install_requires = [line.strip() for line in req_path.read_text().splitlines() if line.strip() and not line.strip().startswith("#")]

setup(
    install_requires=install_requires,
)

