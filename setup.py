#!/usr/bin/env python3
import os
from setuptools import setup, find_packages


def readme():
   with open("README.md") as f:
       return f.read()



with open("requirements.txt", "r") as r:
   filtered_lines = filter(lambda line: not line.startswith("#"), r.readlines())
   requirements = list(map(lambda s: s.replace("\n", ""), filtered_lines))
   print(requirements)


all_packages = find_packages()
setup(
    name="XS3D",
    version="1.0.4",
    description="A Python tool for modeling circular and non-ciruclar motions via emission lines on datacubes.",
    long_description=readme(),
    keywords="kinematics",
    url="https://github.com/CarlosCoba/XS3D",
    author="C. Lopez-Coba",
    author_email="calopez@asiaa.sinica.edu.tw",
	package_data={ '': ['*.txt', '*.ini']},
    license="MIT",
    packages=all_packages,
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'XS3D=xs3d.src.XookSuut3D_inputs:input_params'
        ]
    }
)
