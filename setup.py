from setuptools import find_packages, setup
import platform

with open("README.md", "r", errors='ignore') as f:
    long_description = f.read()

with open('requirements.txt', 'r', encoding='utf-16', errors='ignore') as ff:
    required = ff.read().splitlines()

if platform.system() == 'Windows':
    required.append('raidionicsseg @ git+https://github.com/dbouget/raidionics-seg-lib.git@v0.1.0-alpha')
    # required.append('antspyx @ git+https://github.com/SGotla/ANTsPy/releases/download/0.1.7Win64/antspy-0.1.7-cp37-cp37m-win_amd64.whl')
    required.append('antspy @ git+https://github.com/SGotla/ANTsPy.git@0.1.7Win64')
elif platform.system() == 'Darwin':
    required.append('raidionicsseg @ git+https://github.com/dbouget/raidionics-seg-lib.git@v0.1.0-alpha')
    # required.append('antspyx @ git+https://github.com/ANTsX/ANTsPy/releases/download/v0.1.8/antspyx-0.1.8-cp37-cp37m-macosx_10_14_x86_64.whl')
    required.append('antspyx @ git+https://github.com/ANTsX/ANTsPy.git@v0.2.0')
else:
    required.append('raidionicsseg @ git+https://github.com/dbouget/raidionics-seg-lib.git@v0.1.0-alpha')
    required.append('antspyx')

setup(
    name='raidionicsrads',
    packages=find_packages(
        include=[
            'raidionicsrads',
            'raidionicsrads.Utils',
            'raidionicsrads.Processing',
            'raidionicsrads.NeuroDiagnosis',
            'raidionicsrads.MediastinumDiagnosis',
            'Atlases',
        ]
    ),
    entry_points={
        'console_scripts': [
            'raidionicsrads = raidionicsrads.__main__:main'
        ]
    },
    install_requires=required,
    # include_package_data=True,
    python_requires=">=3.6, <3.8",
    version='0.1.0',
    author='David Bouget (david.bouget@sintef.no)',
    license='BSD 2-Clause',
    description='Raidionics reporting and data system (RADS)',
    long_description=long_description,
    long_description_content_type="text/markdown",
)
