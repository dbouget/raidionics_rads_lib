from setuptools import find_packages, setup
import platform
import sys

with open("README.md", "r", errors='ignore') as f:
    long_description = f.read()

with open('requirements.txt', 'r', encoding='utf-16', errors='ignore') as ff:
    required = ff.read().splitlines()

if platform.system() == 'Windows':
    required.append('antspy@https://github.com/SGotla/ANTsPy/releases/download/0.1.7Win64/antspy-0.1.7-cp37-cp37m-win_amd64.whl')
    # required.append('pandas==1.3.5')
    required.append('scikit-learn==1.0.2')
    required.append('statsmodels==0.13.2')
elif platform.system() == 'Darwin':
    required.append('antspyx@https://github.com/ANTsX/ANTsPy/releases/download/v0.1.8/antspyx-0.1.8-cp37-cp37m-macosx_10_14_x86_64.whl')
    # required.append('pandas==1.3.5')
    required.append('scikit-learn==1.0.2')
    required.append('statsmodels==0.13.2')
    required.append('MedPy==0.4.0')
else:
    required.append('antspyx')

required.append('raidionicsseg@git+https://github.com/dbouget/raidionics_seg_lib.git@master#egg=raidionicsseg')

if sys.version_info >= (3, 8):  # Haven't checked all, but at least 3.8
    required.append('numpy==1.23.1')  # From version 1.24.0 and above, np.bool has been removed, which breaks hd95 computation in MedPy v0.4.0 atm...

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
    python_requires="==3.7",
    version='1.1.0',
    author='David Bouget (david.bouget@sintef.no)',
    license='BSD 2-Clause',
    description='Raidionics reporting and data system backend (RADS)',
    long_description=long_description,
    long_description_content_type="text/markdown",
)
