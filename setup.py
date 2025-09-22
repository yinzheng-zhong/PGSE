from setuptools import setup, find_packages
import pathlib

# Read the contents of requirements.txt
here = pathlib.Path(__file__).parent.resolve()
with open(here / "requirements.txt", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="pgse",
    version="0.8.5",
    author="Yinzheng Zhong",
    author_email="yinzheng.zhong@liverpool.ac.uk",
    description="Progressive Genome Segment Enhancement (PGSE)",
    license="PolyForm Noncommercial License 1.0.0",
    license_files=["LICENSE.md"],
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yinzheng-zhong/pgse",
    packages=find_packages(include=["pgse", "pgse.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    package_data={'pgse': [
        'c_lib/aho_corasick.so',
        'c_lib/aho_corasick.dylib',
        'c_lib/aho_corasick.dll'
    ]},
    entry_points={
        'console_scripts': [
            'pgse-train = pgse.cli:train',
            'pgse-predict = pgse.cli:predict',
        ],
    },
)