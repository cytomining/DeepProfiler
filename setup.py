import setuptools


setuptools.setup(
    name="deepprofiler",
    version="0.3.1",
    author="Juan C. Caicedo",
    author_email="jcaicedo@broad.mit.edu",
    description=("Tools for representation learning in high throughput image collections"),
    license="BSD",
    keywords="",
    url="https://github.com/cytomining/DeepProfiler",
    packages=["deepprofiler"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License"
    ],
    install_requires=[
        "beautifulsoup4>=4.6",
        "click>=6.7",
        "comet_ml>=1.0",
        "efficientnet==1.1.1",
        "gpyopt>=1.2",
        "imagecodecs",
        "lxml>=4.2",
        "numpy>=1.13",
        "pandas>=0.23.0",
        "scikit-image>=0.14.0",
        "scikit-learn>=0.19.0",
        "scipy>=1.1",
        "comet-ml>=3.1.6",
        "tensorflow==2.5.*",
        "tensorflow_addons==0.13.0",
        "tqdm>=4.62",
    ],
    extras_require={
        "test": [
            "pytest>=3.6",
            "pytest-cov>=2.0",
            "codecov>=2.0"
        ]
    }
)