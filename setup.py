import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="not-tf-opt",
    version="0.0.2",
    author="Gergely Flamich",
    author_email="flamich.gergely@gmail.com",
    description="A package that provides a nicer interface to various TF 2.0 optimizers.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gergely-flamich/not-tf-opt",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "tensorflow-probability",
    ]
)
