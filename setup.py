from setuptools import setup


setup(
    name="liveprint",
    version="0.1.0",
    description="Python utility library for dynamic animations projections",
    classifiers=[
        "Topic :: Utilities",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.7",
    ],
    keywords="projections animation projector",
    url="http://github.com/monomonedula/liveprint",
    author="Vladyslav Halchenko",
    author_email="valh@tuta.io",
    license="Apache License Version 2.0",
    packages=["liveprint"],
    test_suite="nose.collector",
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    install_requires=["methodtools"]
)
