from setuptools import setup, find_packages

setup(
    name="pgse",
    version="0.1.0",
    author="Yinzheng Zhong",
    author_email="yinzheng.zhong@liverpool.ac.uk",
    description="Progressive Enhancement of Genome Sequences (PGSE)",
    license_files="LICENSE",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yinzheng-zhong/pgse",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9"
)
