# MCVAE / sMCVAE

Code related to the paper:

*Sparse Multi-Channel Variational Autoencoder for the Joint Analysis of Heterogeneous Data*

by  Luigi Antelmi, Nicholas Ayache, Philippe Robert, Marco Lorenzi.

Link: http://proceedings.mlr.press/v97/antelmi19a.html

BibTeX citation:
```bibtex
@inproceedings{Antelmi2019,
author = {Antelmi, Luigi and Ayache, Nicholas and Robert, Philippe and Lorenzi, Marco},
booktitle = {Proceedings of the 36th International Conference on Machine Learning},
editor = {Chaudhuri, Kamalika and Salakhutdinov, Ruslan},
pages = {302--311},
publisher = {PMLR},
title = {{Sparse Multi-Channel Variational Autoencoder for the Joint Analysis of Heterogeneous Data}}, 
year = {2019}
}
```

# Installation

## GNU/Linux
Install conda:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  
bash Miniconda3-latest-Linux-x86_64.sh
```

Download this github repository and move into in:
```bash
git clone git://gitlab.inria.fr/epione_ML/mcvae
cd mcvae
```

Install the customized python environment:
```bash
conda env create -f environment.yml
```

Activate the python environment:
```bash
conda activate mcvae
```

Install the mcvae package:
```bash
python setup.py install
```

An alternative to the last point is to install the package in "develop" mode.
Using this mode, all local modifications of source code will be considered in your Python interpreter (when restarted) without having to install the package again.
This is particularly useful when adding new features.
To install this package in develop mode, type the following command line:
```bash
python setup.py develop
```

## Windows
Download and install conda from: https://docs.conda.io/en/latest/miniconda.html

Download this github repository from: https://gitlab.inria.fr/epione_ML/mcvae

Open the Anaconda prompt and move into the github repository previously downloaded.

Deactivate the base environment:
`conda deactivate`

Install the customized python environment:
`conda env create -f environment.yml`

Activate the python environment:
`conda activate mcvae`

Install the mcvae package:
`python setup.py install`

An alternative to the last point is to install the package in "develop" mode.
Using this mode, all local modifications of source code will be considered in your Python interpreter (when restarted) without having to install the package again.
This is particularly useful when adding new features.
To install this package in develop mode, type the following command line:
`python setup.py develop`
