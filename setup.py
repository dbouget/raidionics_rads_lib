from setuptools import find_packages, setup
import platform
import sys

with open("README.md", "r", errors='ignore') as f:
    long_description = f.read()

with open('requirements.txt', 'r', encoding='utf-16', errors='ignore') as ff:
    required = ff.read().splitlines()

if platform.system() == 'Windows':
    required.append('antspyx@https://github.com/ANTsX/ANTsPy/releases/download/v0.3.8/antspyx-0.3.8-cp38-cp38-win_amd64.whl')
elif platform.system() == 'Darwin' and platform.processor() == 'arm':   # Specific for Apple M1 chips
    required.append('scikit-learn==1.3.0')
    required.append('statsmodels==0.14.0')
elif platform.system() == 'Darwin':
    required.append('antspyx@https://github.com/ANTsX/ANTsPy/releases/download/v0.3.8/antspyx-0.3.8-cp38-cp38-macosx_10_9_x86_64.whl')
else:
    required.append('antspyx')

required.append('raidionicsseg@git+https://github.com/dbouget/raidionics_seg_lib.git@update#egg=raidionicsseg')

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
            'raidionicsrads.Atlases',
            'tests',
        ]
    ),
    entry_points={
        'console_scripts': [
            'raidionicsrads = raidionicsrads.__main__:main'
        ]
    },
    install_requires=required,
    include_package_data=True,
    python_requires=">=3.7",
    version='1.1.1',
    author='David Bouget (david.bouget@sintef.no)',
    license='BSD 2-Clause',
    description='Raidionics reporting and data system backend (RADS)',
    long_description=long_description,
    long_description_content_type="text/markdown",
)
