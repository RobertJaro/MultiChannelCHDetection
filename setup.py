import setuptools
from setuptools import setup

setup(
    name='MultiChannelCHDetection',
    version='0.2',
    packages=setuptools.find_packages(),
    url='https://github.com/RobertJaro/MultiChannelCHDetection',
    license='GPL-3.0',
    author='Robert Jarolim',
    author_email='',
    description='CHRONNOS',
    install_requires=['torch>=1.8', 'sunpy>=3.0', 'tqdm', 'drms', 'astropy'],
)
