import pathlib
from setuptools import setup, find_packages

# Read the contents of requirements.txt
here = pathlib.Path(__file__).parent.resolve()
with open(here / "requirements.txt", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="pgse",
    version="0.3.4",
    author="Yinzheng Zhong",
    author_email="yinzheng.zhong@liverpool.ac.uk",
    description="Progressive Genome Sequence Enhancement (PGSE)",
    license_files="LICENSE",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yinzheng-zhong/pgse",
    packages=find_packages(include=["pgse", "pgse.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
)