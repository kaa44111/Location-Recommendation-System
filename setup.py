from setuptools import setup, find_packages

setup(
    name='FinalAssignment',
    version='1.0',
    packages=find_packages(include=['src', 'src.*']),
    install_requires=['pandas', 'numpy', 'scikit-learn', 'folium', 'geopy'],
)
