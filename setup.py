import setuptools
import pkg_resources

with open("requirements.txt", "r") as f:
    requirements = [str(req) for req in pkg_resources.parse_requirements(f)]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sklearn_cv_pandas",
    version="0.0.5",
    author="@not-so-fat",
    description="RandomizedSearchCV/GridSearchCV with pandas.DataFrame interface",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/not-so-fat/sklearn_cv_pandas",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
