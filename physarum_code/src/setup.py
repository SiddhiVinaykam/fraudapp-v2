
from setuptools import setup, find_packages
import ast

NAME = "fraudappv2"
VERSION = "0.0.1"
REQUIRES = ['click', 
'kfp', 
'xgboost', 
'pandas', 
'xgboost', 
'google-cloud', 
'google-cloud-bigquery', 
'physarum', 
'pytest', 
'httplib2', 
'sklearnPipeline'
]

setup(
    name=NAME,
    version=VERSION,
    install_requires=REQUIRES,
    packages=find_packages(),
    python_requires=">=3.5.3",
    include_package_data=True,
    entry_points={"console_scripts": ["fraudappv2 = fraudappv2.start:main"]},
)
