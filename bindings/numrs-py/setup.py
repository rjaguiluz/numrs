
from setuptools import setup, find_packages

setup(
    name="numrs",
    version="0.1.0",
    description="Python bindings for NumRs",
    packages=find_packages(),
    install_requires=[
        "typing_extensions; python_version < '3.8'",
    ],
    python_requires=">=3.7",
    package_data={"numrs": ["*.so", "*.dylib", "*.dll"]},
    include_package_data=True,
    license="AGPL-3.0-only",
)
