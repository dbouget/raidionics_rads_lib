# Raidionics backend for computing tumor characteristics and generating a standardized report (RADS).

[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
#[![Build Actions Status](https://github.com/dbouget/raidionics-rads-lib/workflows/Build/badge.svg)](https://github.com/dbouget/raidionics-rads-lib/actions)
[![Paper](https://zenodo.org/badge/DOI/10.48550/arXiv.2204.14199.svg)](https://doi.org/10.48550/arXiv.2204.14199)

The code corresponds to the RADS backend of MRI/CT volumes.  
The module can either be used as a Python library, as CLI, or as Docker container.

# Installation

```
pip install git+https://github.com/dbouget/raidionics-rads-lib.git
```

# Usage
## CLI
```
raidionicsseg CONFIG
```

CONFIG should point to a configuration file (*.ini), specifying all runtime parameters,
according to the pattern from [**blank_main_config.ini**](https://github.com/dbouget/raidionics-rads-lib/blob/master/blank_main_config.ini).

## Python module
```
from raidionicsrads import run_rads
run_rads(config_filename="/path/to/main_config.ini")
```

## Docker
```
docker pull dbouget/raidionics-rads:v1
docker run --entrypoint /bin/bash -v /home/ubuntu:/home/ubuntu -t -i --runtime=nvidia --network=host --ipc=host dbouget/raidionics-radsr:v1
```

The `/home/ubuntu` before the column sign has to be changed to match your local machine.

# Models
The trained models, necessary for segmentation, are automatically downloaded when running Raidionics or Raidionics-Slicer.

# How to cite
Please, consider citing our paper, if you find the work useful:

```
@misc{https://doi.org/10.48550/arXiv.2204.14199,
title = {Preoperative brain tumor imaging: models and software for segmentation and standardized reporting},
author = {Bouget, D. and Pedersen, A. and Jakola, A. S. and Kavouridis, V. and Emblem, K. E. and Eijgelaar, R. S. and Kommers, I. and Ardon, H. and Barkhof, F. and Bello, L. and Berger, M. S. and Nibali, M. C. and Furtner, J. and Hervey-Jumper, S. and Idema, A. J. S. and Kiesel, B. and Kloet, A. and Mandonnet, E. and Müller, D. M. J. and Robe, P. A. and Rossi, M. and Sciortino, T. and Brink, W. Van den and Wagemakers, M. and Widhalm, G. and Witte, M. G. and Zwinderman, A. H. and Hamer, P. C. De Witt and Solheim, O. and Reinertsen, I.},
doi = {10.48550/ARXIV.2204.14199},
url = {https://arxiv.org/abs/2204.14199},
keywords = {Image and Video Processing (eess.IV), Computer Vision and Pattern Recognition (cs.CV), Machine Learning (cs.LG), FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Computer and information sciences, FOS: Computer and information sciences, I.4.6; J.3},
publisher = {arXiv},
year = {2022},
copyright = {Creative Commons Attribution 4.0 International}}
```
