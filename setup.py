import setuptools


setuptools.setup(
    name="deep-profiler",
    version="0.1.0",
    author="Juan Caicedo",
    author_email="jcaicedo@gmail.com",
    description=("Tools for representation learning in high throughput image collections"),
    license="BSD",
    keywords="",
    url="https://github.com/jccaicedo/DeepProfiler",
    packages=["deepprofiler"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License"
    ],
    install_requires=[
        "click>=6.7",
        "cycler>=0.10.0",
        "decorator>=4.1.2",
        "matplotlib>=2.0.2",
        "networkx>=1.11",
        "numpy>=1.13.1",
        "olefile>=0.44",
        "pandas>=0.20.3",
        "Pillow>=4.2.1",
        "pyparsing>=2.2.0",
        "python-dateutil>=2.6.1",
        "pytz>=2017.2",
        "PyWavelets>=0.5.2",
        "scikit-image>=0.13.0",
        "scipy>=0.19.1",
        "six>=1.10.0",
        "tensorflow>=1.8.0",
        "comet_ml>=1.0.0",
        "keras>=2.2.0",
        "keras_resnet>=0.1.0",
        "scikit-learn>=0.19.1",
        "gpyopt>=1.2.5"
    ],
    setup_requires=["pytest-runner"],
    tests_requires=["pytest"]
)
