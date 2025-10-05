#!/usr/bin/env python3
"""
Setup script for DGA Domain Classifier.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
with open(readme_path, 'r', encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
with open(req_path, 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='dga-classifier',
    version='1.0.0',
    description='High-performance DGA domain classifier with incremental learning',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='DGA Classifier Team',
    author_email='team@dgaclassifier.com',
    url='https://github.com/your-org/dga-classifier',
    
    packages=find_packages(),
    
    install_requires=requirements,
    
    extras_require={
        'performance': ['dask[complete]'],
        'dev': ['pytest', 'black', 'flake8', 'mypy'],
    },
    
    entry_points={
        'console_scripts': [
            'dga-train=main:main',
            'dga-predict=predict:main',
            'dga-demo=example_usage:main',
        ],
    },
    
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Security',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    
    python_requires='>=3.8',
    
    keywords='dga domain classification machine-learning security cybersecurity',
    
    project_urls={
        'Bug Reports': 'https://github.com/your-org/dga-classifier/issues',
        'Source': 'https://github.com/your-org/dga-classifier',
        'Documentation': 'https://github.com/your-org/dga-classifier/wiki',
    },
)