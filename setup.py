from setuptools import setup, find_packages
from typing import List
import os

HYPEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> List[str]:
    """
    This function will return the list of requirements from a requirements.txt file.
    """
    requirements = []
    with open(file_path, "r", encoding="utf-8") as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements]  # remove \n and extra spaces

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

# Get requirements from file
requirements = get_requirements("requirements.txt")

setup(
    name="cyberbullying_detector",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A fine-tuned LLM for detecting cyberbullying in text",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/bassemdahmeni/End-to-End-ML-Application-for-Cyber_Bullying_Detection",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "cyberbully-predict=cyberbullying_detector.cli:main"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
    ],
)
