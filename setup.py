from setuptools import setup, find_packages

setup(
    name="parsepro",
    version="0.1.0",
    description="A Python library to parse documents to Markdown format .",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Jobanpreet Singh",
    author_email="singhjoban495@gmail.com",
    url="https://github.com/jobanpreet/parsepro",
    packages=find_packages(),
    install_requires=[
        "together",  # List all required dependencies
    ],
    extras_require={
        "dev": ["pytest", "flake8"],  # Development dependencies
    },
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    project_urls={
        "Documentation": "https://github.com/jobanpreet/metaocr#readme",
        "Source Code": "https://github.com/jobanpreet/metaocr",
    },
)
