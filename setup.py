from setuptools import find_packages, setup
import platform
import sys

with open("README.md", "r", errors='ignore') as f:
    long_description = f.read()

with open('requirements.txt', 'r', encoding='utf-16', errors='ignore') as ff:
    required = ff.read().splitlines()

setup(
    name='raidionicsrads',
    packages=find_packages(
        include=[
            'raidionicsrads',
            'raidionicsrads.Utils',
            'raidionicsrads.Utils.DataStructures',
            'raidionicsrads.Utils.ReportingStructures',
            'raidionicsrads.Processing',
            'raidionicsrads.Pipelines',
            'raidionicsrads.NeuroDiagnosis',
            'raidionicsrads.MediastinumDiagnosis',
            'tests',
        ]
    ),
    entry_points={
        'console_scripts': [
            'raidionicsrads = raidionicsrads.__main__:main'
        ]
    },
    install_requires=required + [
        "scikit-learn; platform_system=='Darwin' and platform_machine=='arm64'",
        "statsmodels; platform_system=='Darwin' and platform_machine=='arm64'",
        "antspyx==0.4.2; platform_system=='Windows' and python_version<'3.10'",
        "antspyx==0.6.1; platform_system=='Windows' and python_version>='3.10'",
        "antspyx==0.6.1; platform_system!='Windows' and platform_machine!='arm64'",
    ],
    include_package_data=True,
    package_data={
        "raidionicsrads": [
            "Atlases/**/*",  # all files in Atlases/
        ],
    },
    python_requires=">=3.9",
    version='1.3.1',
    author='David Bouget (david.bouget@sintef.no)',
    license='BSD 2-Clause',
    description='Raidionics reporting and data system backend (RADS)',
    long_description=long_description,
    long_description_content_type="text/markdown",
)
