# +
from distutils.core import setup


setup(
    name="easy-meta-md",
    version="v0",
    author="Amir HajiBabaei T.",
    author_email="a.hajibabaei.86@gmail.com",
    description="symbolic definition of collective variables and automation of metadynamics",
    url="https://github.com",
    package_dir={'emeta': 'emeta'},
    packages=['emeta'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
